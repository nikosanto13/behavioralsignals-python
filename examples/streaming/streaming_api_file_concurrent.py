import os
import json
import time
import argparse
import threading
from typing import Any, Literal, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv

from behavioralsignals import Client, StreamingOptions
from behavioralsignals.utils import make_audio_stream


def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavioral Signals API Client Example (concurrent streaming benchmark)"
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the audio file to send"
    )
    parser.add_argument(
        "--batch_file_path", type=str, default=None, help="Path to the audio file to use for concurrent batch processing (if different from --file_path)"
    )
    parser.add_argument(
        "--enable_batch_worker",
        action="store_true",
        help="Run a single background batch upload+poll worker alongside streaming (benchmarking)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="behavioral",
        choices=["behavioral", "deepfakes"],
        help="API to use for streaming",
    )
    parser.add_argument(
        "--response_level",
        type=str,
        default="segment",
        choices=["segment", "utterance", "all"],
        help="Level of response granularity",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel streaming sessions to run",
    )

    return parser.parse_args()


def _run_one_session(
    *,
    session_id: int,
    cid: str,
    api_key: str,
    api: str,
    response_level: Literal["segment", "utterance", "all"],
    file_path: str,
) -> dict:
    start = time.monotonic()

    with Client(cid=cid, api_key=api_key) as client:
        audio_stream = make_audio_stream(
            file_path, chunk_size=0.25, realtime_pacing=True
        )
        options = StreamingOptions(
            sample_rate=audio_stream.sample_rate,
            encoding="LINEAR_PCM",
            level=response_level,
        )

        if api == "behavioral":
            responses = client.behavioral.stream_audio(audio_stream=audio_stream, options=options)
        else:
            responses = client.deepfakes.stream_audio(audio_stream=audio_stream, options=options)

        output_data = []
        print(f"Session {session_id} started streaming")
        for resp in responses:
            output_data.append(resp.model_dump())
        print(f"Session {session_id} completed streaming")

    duration_s = time.monotonic() - start
    return {
        "session_id": session_id,
        "duration_s": duration_s,
        "responses": output_data,
        "responses_count": len(output_data),
    }


def _poll_process_until_done(*, client, pid: int, stop_event: threading.Event) -> tuple[object, bool]:
    last_status = None
    while True:
        if stop_event.is_set():
            return None, True

        process = client.get_process(pid=pid)
        status = process.statusmsg

        if process.is_completed:
            if last_status != process.statusmsg:
                print(f"[batch] pid={pid} completed")
            return process, False
        if process.is_processing:
            if last_status != process.statusmsg:
                print(f"[batch] pid={pid} processing")
        elif process.is_pending:
            if last_status != process.statusmsg:
                print(f"[batch] pid={pid} pending (API busy)")
        else:
            if last_status != process.statusmsg:
                print(f"[batch] pid={pid} unexpected status: {process.statusmsg}")
            return process, False

        last_status = status
        time.sleep(1.0)


def _run_batch_worker(
    *,
    cid: str,
    api_key: str,
    api: str,
    file_path: str,
    iterations: int,
    stop_event: threading.Event,
) -> dict:
    start = time.monotonic()
    runs: list[dict] = []

    with Client(cid=cid, api_key=api_key) as client:
        batch_client = client.behavioral if api == "behavioral" else client.deepfakes

        for i in range(iterations):
            if stop_event.is_set():
                break

            iter_start = time.monotonic()
            try:
                print(f"[batch] iteration={i} uploading...")
                upload_response = batch_client.upload_audio(file_path=file_path)
                pid = upload_response.pid
                print(f"[batch] iteration={i} uploaded pid={pid}")
            except Exception as e:
                runs.append(
                    {
                        "iteration": i,
                        "pid": None,
                        "completed": False,
                        "statusmsg": None,
                        "duration_s": time.monotonic() - iter_start,
                        "error": repr(e),
                    }
                )
                continue

            process, stopped = _poll_process_until_done(
                client=batch_client, pid=pid, stop_event=stop_event
            )

            runs.append(
                {
                    "iteration": i,
                    "pid": pid,
                    "completed": (process is not None and getattr(process, "is_completed", False) and not stopped),
                    "statusmsg": (None if process is None else getattr(process, "statusmsg", None)),
                    "duration_s": time.monotonic() - iter_start,
                    "stopped_early": stopped,
                }
            )

    duration_s = time.monotonic() - start
    completed = sum(1 for r in runs if r.get("completed"))
    return {
        "iterations_target": iterations,
        "iterations_started": len(runs),
        "iterations_completed": completed,
        "duration_s": duration_s,
        "runs": runs,
    }


if __name__ == "__main__":
    args = parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")

    load_dotenv()
    cid = os.getenv("CID")
    api_key = os.getenv("API_KEY")
    if cid is None or api_key is None or cid == "" or api_key == "":
        raise SystemExit("Missing CID/API_KEY in environment (set them or provide a .env)")
    overall_start = time.monotonic()

    stop_event = threading.Event()
    batch_result_holder: dict[str, Any] = {"value": None}
    batch_thread: threading.Thread | None = None

    if args.enable_batch_worker:
        batch_file_path = (
            args.batch_file_path if args.batch_file_path is not None else args.file_path
        )

        def _batch_runner():
            batch_result_holder["value"] = _run_batch_worker(
                cid=cid,
                api_key=api_key,
                api=args.api,
                file_path=batch_file_path,
                iterations=args.concurrency,
                stop_event=stop_event,
            )

        batch_thread = threading.Thread(target=_batch_runner, name="batch-worker", daemon=True)
        batch_thread.start()

    sessions = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    _run_one_session,
                    session_id=i,
                    cid=cid,
                    api_key=api_key,
                    api=args.api,
                    response_level=cast(Literal["segment", "utterance", "all"], args.response_level),
                    file_path=args.file_path,
                )
                for i in range(args.concurrency)
            ]

            for future in as_completed(futures):
                sessions.append(future.result())
    finally:
        stop_event.set()
        if batch_thread is not None:
            batch_thread.join(timeout=10.0)

    overall_duration_s = time.monotonic() - overall_start
    latencies = []
    for session in sessions:
        for resp in session["responses"]:
            duration = float(resp["results"][0]["endTime"]) - float(resp["results"][0]["startTime"])
            # NOTE: Do not consider responses with duration = 1.0s (durations can be either 1.0 or 2.0)
            # since these will be always delayed by at least 1.0s
            if duration > 1.0:
                latencies.append(resp["delay_ms"])

    # calculate mean, p50, p90, p95, p99 latencies
    latency_ms = {
        "mean": np.mean(latencies),
        "p50": np.percentile(latencies, 50),
        "p90": np.percentile(latencies, 90),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
    }

    output_payload = {
        "api": args.api,
        "response_level": args.response_level,
        "file_path": args.file_path,
        "concurrency": args.concurrency,
        "overall_duration_s": overall_duration_s,
        "sessions": sorted(sessions, key=lambda x: x["session_id"]),
        "latency_ms": latency_ms,
        "batch_worker": batch_result_holder["value"],
    }

    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=4)

    print(f"Latency (ms) report: {latency_ms}")
    print(f"Results saved to {args.output}")

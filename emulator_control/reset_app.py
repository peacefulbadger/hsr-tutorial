"""Force-stop / relaunch helpers for the emulated client.

Phase 0, item 7 of the plan. Kept separate from the sidecar so the sidecar stays
a pure camera+finger, but it talks to the same adb instance.
"""

from __future__ import annotations

import argparse
import time

from sidecar_windows.adb import AdbClient, AdbConfig, AdbError

DEFAULT_PACKAGE = "com.HoYoverse.hkrpgoversea"
LAUNCHER_COMPONENT = "com.mihoyo.combosdk.ComboSDKActivity"


class EmulatorController:
    def __init__(self, adb: AdbClient, package: str = DEFAULT_PACKAGE) -> None:
        self.adb = adb
        self.package = package

    def is_running(self) -> bool:
        return self.adb.package_running(self.package)

    def force_stop(self) -> None:
        self.adb.shell("am", "force-stop", self.package)

    def launch(self) -> None:
        # monkey resolves the launcher intent without us hardcoding a component
        # that drifts between bundles.
        self.adb.shell(
            "monkey",
            "-p", self.package,
            "-c", "android.intent.category.LAUNCHER",
            "1",
        )

    def reset_app(self, *, wait_running_s: float = 30.0, settle_s: float = 0.0) -> dict:
        started = time.time()
        was_running = self.is_running()
        self.force_stop()
        time.sleep(0.5)
        self.launch()

        deadline = time.monotonic() + wait_running_s
        running = False
        while time.monotonic() < deadline:
            if self.is_running():
                running = True
                break
            time.sleep(0.25)
        if settle_s:
            time.sleep(settle_s)
        return {
            "ok": running,
            "package": self.package,
            "was_running": was_running,
            "running_after": running,
            "elapsed_s": round(time.time() - started, 2),
            "focus": self.adb.current_focus() if running else "",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset (force-stop + relaunch) the emulated client")
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--serial", default="")
    parser.add_argument("--settle", type=float, default=0.0, help="extra seconds to wait after launch")
    args = parser.parse_args(argv)

    try:
        adb = AdbClient(AdbConfig(adb_path=args.adb_path, serial=args.serial))
    except AdbError as exc:
        print(f"error: {exc}")
        return 1

    controller = EmulatorController(adb, package=args.package)
    result = controller.reset_app(settle_s=args.settle)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

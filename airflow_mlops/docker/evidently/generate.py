from app import build_report
if __name__ == "__main__":
    p = build_report()
    print(f"[evidently] wrote {p}")
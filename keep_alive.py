import time

def keep_alive(interval_minutes=5):
    print(f"💡 Terminal keep-alive every {interval_minutes} minutes.")
    print("⏹️ Press Ctrl+C to stop.\n")

    try:
        while True:
            print(f"✅ Still alive at {time.strftime('%H:%M:%S')}")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("\n🛑 Script stopped by user.")

if __name__ == "__main__":
    keep_alive()
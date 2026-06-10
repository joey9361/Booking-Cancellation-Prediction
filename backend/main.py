from src.testing_database import load_query
def main():
    # Load the data
    df = load_query(datamanager, "SELECT * FROM bookings")
    pass

if __name__ == "__main__":
    from src.testing_database import create_datamanager
    datamanager = create_datamanager()
    main()
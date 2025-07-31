import pandas as pd

def import_sheet(sheet_url: str) -> pd.DataFrame:
    """
    Imports a Google Sheet into a pandas DataFrame.

    This function takes a URL to a Google Sheet, processes it, and retrieves the data
    as a pandas DataFrame. It assumes that the sheet is publicly accessible or has
    the necessary sharing permissions enabled to access its content as a CSV file.

    :param sheet_url: The URL of the Google Sheet to import.
    :type sheet_url: str
    :return: The data contained in the sheet represented as a pandas DataFrame.
    :rtype: pd.DataFrame
    """
    base_url = sheet_url.split("/edit")[0]
    return pd.read_csv(f"{base_url}/export?format=csv&gid=0")

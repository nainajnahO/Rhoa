# rhoa - A pandas DataFrame extension for technical analysis
# Copyright (C) 2025 nainajnahO
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

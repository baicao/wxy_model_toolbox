import pandas as pd


def load_2_dataframe(file_name) -> pd.DataFrame:
    df = None
    if file_name.endswith("xlsx"):
        df = pd.read_excel(file_name)
    elif file_name.endswith("feather"):
        df = pd.read_feather(file_name)
    elif file_name.endswith("csv"):
        df = pd.read_csv(file_name, sep="\001")
    return df


def dataframe_save(df: pd.DataFrame,
                   file_name: str,
                   extensions: str = None) -> str:
    if extensions is None:
        extensions = file_name.split(".")[-1]
        if extensions not in ["feather", "xlsx", "csv"]:
            return None
        else:
            file_name = "{}.{}".format(file_name, extensions)
    if extensions == "feather":
        df.to_feather(file_name)
    elif extensions == "xlsx":
        df.to_excel(file_name, index=False)
    elif extensions == "csv":
        df.to_csv(file_name, sep="\001", index=False)
    else:
        return None
    return file_name

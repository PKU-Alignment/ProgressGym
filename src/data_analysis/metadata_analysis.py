# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from src import text_writer
import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
from collections import defaultdict
from collections import Counter
import glob
import pandas as pd


# %%
def metadata_desc(lst_content_length, lst_source_dataset, lst_culture, lst_language):
    """
    Runs exploratory data analyses on all metadata and prints results.
    """
    # summary statistics
    num_docs = len(lst_content_length)
    summary_stats = pd.DataFrame(
        {
            "Number of Documents": num_docs,
            "Average Document Length": sum(lst_content_length) / num_docs,
            "Std": np.std(lst_content_length),
            "Max Length": max(lst_content_length),
            "Min Length": min(lst_content_length),
        },
        index=[0],
    )
    print(summary_stats)

    # frequency counts
    source_dataset_counts = Counter(lst_source_dataset)
    culture_counts = Counter(lst_culture)
    language_counts = Counter(lst_language)

    print("Number of Documents by Source Dataset:")
    for source, count in source_dataset_counts.items():
        print(f"{source}: {count}")

    print("\nNumber of Documents by Culture:")
    for source, count in culture_counts.items():  # issues with same source
        source = source.strip("\n")
        print(f"{source}: {count}")

    print("\nNumber of Documents by Language:")
    for source, count in language_counts.items():
        print(f"{source}: {count}")


# %%
def length_over_time(years, lengths):
    """
    Runs linear regression on document content length over time.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, lengths)
    print(
        f"Slope: {slope}, Intercept: {intercept}, R-value: {r_value}, P-value: {p_value}, Std Err: {std_err}"
    )

    plt.scatter(years, lengths, alpha=0.5)
    plt.plot(years, [slope * x + intercept for x in years], color="red")
    plt.title("Content Length over Time")
    plt.xlabel("Year")
    plt.ylabel("Content Length")
    plt.show()


def year_to_century(year):
    """
    Helper method for `aggregate_docs_by_century`.
    """
    if year > 0:
        return (year - 1) // 100 + 1
    else:  # case of BC
        return (year // 100) - 1


def aggregate_docs_by_century(lst_creation_year):
    """
    Helper method for `num_docs_over_time`.
    """
    century_counts = defaultdict(int)
    for year in lst_creation_year:
        if year != -1:  # missing data
            century = year_to_century(year)
            century_counts[century] += 1
    return century_counts


def num_docs_over_time(century_counts):
    """
    Runs linear regression on number of documents over time (in centuries).
    """
    centuries = sorted(century_counts.keys())
    doc_counts = [century_counts[century] for century in centuries]

    X = np.array(centuries)
    Y = np.array(doc_counts)

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print(
        f"Slope: {slope}, Intercept: {intercept}, R-value: {r_value}, P-value: {p_value}, Std Err: {std_err}"
    )

    plt.figure(figsize=(10, 6))
    plt.bar(X, Y, color="skyblue", label="Number of Documents")
    plt.plot(X, slope * X + intercept, color="red", label="Trend line", linewidth=2)
    plt.title("Number of Documents by Century")
    plt.xlabel("Century")
    plt.xticks(X, labels=[str(int(x)) for x in X], rotation=45)
    plt.ylabel("Number of Documents")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
def main():
    """
    Reads in complete dataset and runs EDA and linear regressions on metadata.
    """
    # initialize fields used for metadata collection
    lst_content_length = []
    lst_creation_year = []
    lst_creation_year_earliest = []
    lst_creation_year_latest = []
    lst_source_dataset = []
    lst_source_dataset_detailed = []
    lst_culture = []
    lst_language = []

    # iteratively read in all json files in directory
    for pathname in tqdm(
        glob.iglob(
            "dataset/dataset_text_sequence/histext_1826_to_2018_collection_G/**/**/*.json",
            recursive=True,
        )
    ):
        dict_iterator = text_writer.read_json_memory_efficient(pathname)
        for doc in tqdm(dict_iterator):
            # extract metadata
            lst_content_length.append(len(doc.get("content", "")))
            lst_creation_year.append(doc.get("creation_year", -1))
            lst_creation_year_earliest.append(
                doc.get("creation_year_earliest", -1)
            )  # -1 if data is missing
            lst_creation_year_latest.append(
                doc.get("creation_year_latest", -1)
            )  # -1 if data is missing
            lst_source_dataset.append(doc.get("source_dataset", "None"))
            lst_source_dataset_detailed.append(
                doc.get("source_dataset_detailed", "None")
            )
            lst_culture.append(doc.get("culture", "None"))

            language = doc.get("language", "None")
            if isinstance(language, list):
                language = (
                    language[0] if language else "None"
                )  # only take first language (may not be best approach)
            lst_language.append(language)

    # descriptive analyses
    metadata_desc(lst_content_length, lst_source_dataset, lst_culture, lst_language)

    # linear regression analyses
    length_over_time(np.array(lst_creation_year), np.array(lst_content_length))
    century_counts = aggregate_docs_by_century(lst_creation_year)
    num_docs_over_time(century_counts)
    # %%


main()

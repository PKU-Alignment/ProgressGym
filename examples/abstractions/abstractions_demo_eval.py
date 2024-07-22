from src.abstractions import Model, Data, DataFileCollection
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    c021_instruct = Model("C021-instruct", is_instruct_finetuned=True)
    print(c021_instruct.evaluate())

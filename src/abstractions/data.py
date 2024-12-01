# from src.abstractions.model import Model   # Uncommenting will lead to circular import
from src.path import root
from typing import (
    Dict,
    Any,
    Literal,
    Optional,
    List,
    Callable,
    Hashable,
    Iterable,
    Any,
    Union,
)
import os
import json
import warnings
import src.utils.text_utils as tu
from tqdm import tqdm
from src.abstractions.configs.templates_configs import *


# helper function, escape spaces in paths
def escape(path: str):
    return path.strip().replace(" ", "\\ ")


# helper function, executes a command in terminal
def execute(command: str):
    return_value = os.system(command)
    if return_value:
        warnings.warn(
            f"Command returns non-zero value {return_value}. Command: {command}"
        )


# helper function, removes fields with non-simple types
def clean_dict(d: Dict, filter_fields: Optional[List[str]] = None) -> Dict:
    return {
        k: v
        for k, v in d.items()
        if (filter_fields is None or k in filter_fields)
        and isinstance(v, (int, float, str, bool, type(None)))
    }


class Data:
    """
    The Data class stores a body of data's path, format, relevant fields, etc., allowing for memory-efficient manipulation of large bodies of data.
    """

    # mapping from data name to Data instance (used Any due to typing constraints), updated on the fly
    name2data: Dict[str, Any] = {}
    always_force_rewrite: bool = True
    data_type: Literal["pretrain", "sft", "preference"]
    
    default_key_fields = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "history": "history",
    }

    # check with user before removing a file
    @classmethod
    def ask_and_remove_if_exists(cls, path: str, forced_rewrite: bool):
        if os.path.exists(path):
            if forced_rewrite or (
                hasattr(cls, "always_force_rewrite") and cls.always_force_rewrite
            ):
                execute(f'rm {"-r" if os.path.isdir(path) else ""} -f {escape(path)}')
                return

            warnings.warn(
                f"{path} already exists. Use forced_rewrite=True to force rewrite."
            )
            answer = input("Do you want to force rewrite? (yes/no/always) ").lower()
            if "n" in answer:
                return
            if "a" in answer:
                cls.always_force_rewrite = True
            execute(f'rm {"-r" if os.path.isdir(path) else ""} {escape(path)}')

    def __init__(
        self,
        data_name: str,
        data_type: Literal["pretrain", "sft", "preference"] = "sft",
        data_path: Optional[str] = None,
        data_content: List[Dict] = None,
        **kwargs,
    ):
        """
        Initialize.

        :param data_name: Necessary, name of the data
        :type data_name: str

        :param data_type: Optional, type of usage of data, i.e. which stage of training it will be used in.
        :type data_type: Literal["pretrain", "sft", "preference"] = "sft"

        :param data_path: Optional. Search path of data. When data_path is omitted, make sure it exists in './libs/llama_factory/data/' or other data search paths (see abstractions_config.json) and is recognized by Llama-Factory.
        :type data_path: Optional[str] = None
        :raise FileNotFoundError: If file is not found in default search path and path is not specified.

        :param data_content: Optional. Content of data. When data_content is provided, the given content will be written to a data_path to create a new dataset, unless data_path is not provided in which case the dataset will be saved to './output/datasets/{data_name}.json'.
        :type data_content: List[Dict] = None

        Examples:
            .. code-block:: python

                Data('c4_demo', data_type = 'sft', data_path = './libs/llama_factory/data/c4_demo.json')
                Data('c4_demo', data_type = 'sft')

        """
        # if data_name in Data.name2data:
        #     warnings.warn(f'The data name {data_name} is already in use.')

        if "is_instruction_data" in kwargs:
            warnings.warn(
                f"is_instruction_data is deprecated. Please use data_type instead."
            )
            data_type = "sft" if kwargs["is_instruction_data"] else "pretrain"

        if data_content is not None:
            if data_path is None:
                data_path = f"{root}/output/datasets/{data_name}.json"
                Data.ask_and_remove_if_exists(data_path, forced_rewrite=True)

            with tu.JsonListWriter(data_path) as json_writer:
                for element in data_content:
                    json_writer.append(element)

        self.data_name = data_name
        self.data_path = data_path
        self.data_type = data_type
        self.key_fields = {}

        # if data_path is not specified, look for it in the paths specified in abstractions_config.json
        if not data_path:
            for search_path in data_search_paths:
                if os.path.exists(os.path.join(search_path, data_name + ".json")):
                    print(
                        f'Found data {data_name} at {os.path.join(search_path, data_name + ".json")}'
                    )
                    self.data_path = os.path.join(search_path, data_name + ".json")
                    break

        # data_path not found
        if self.data_path is None:
            print(
                f"Data {data_name} not found locally. Searching among Llama-Factory datasets."
            )
            with open(f"{root}/libs/llama_factory/data/dataset_info.json", "r") as in_file:
                registrations = json.load(in_file)

            if self.data_name in registrations:
                self.data_path = f'{root}/libs/llama_factory/data/{registrations[self.data_name]["file_name"]}'
                print(f'Found {registrations[self.data_name]["file_name"]}.')
            else:
                raise FileNotFoundError(
                    f"The data {self.data_name} doesn't exist either locally or in Llama-Factory registrations."
                )

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The data path {self.data_path} doesn't exist.")

        if data_name in Data.name2data:
            Data.name2data[data_name].append(self)
        else:
            Data.name2data[data_name] = [self]

    def copy(self, data_name: str = None) -> "Data":
        """
        Returns a copy of the current Data instance. 
        Shallow copy if data_name is not provided or identical to the current data_name; deep copy otherwise.
        """
        if data_name and data_name != self.data_name:
            new_data_path = f"{root}/output/datasets/{data_name}.json"
            Data.ask_and_remove_if_exists(new_data_path, forced_rewrite=True)
            execute(f"cp {escape(self.data_path)} {escape(new_data_path)}")
            cp = Data(data_name, self.data_type, new_data_path)
        else:
            cp = Data(self.data_name, self.data_type, self.data_path)
        
        cp.key_fields = self.key_fields.copy()
        return cp

    def transform(
        self,
        transformation: Union[
            Callable[[Dict], Dict], Callable[[List[Dict]], List[Dict]]
        ],
        result_data_name: str,
        forced_rewrite: bool = False,
        max_batch_size: int = 1,
        keep_key_fields: bool = True,
        map_key_fields: bool = False,
    ) -> "Data":
        """
        Apply transformation to every element of the current dataset (in the format of a json list of json dicts where the values are of mutable or immutable types), and returns a Data instance containing the resulting dataset.

        Out-of-place. Does not modify self.

        This function (like all others in abstractions) is memory-efficient for huge json files.
        The data file will be a json file with the type of List[Dict[Hashable, Any]].

        :param transformation: Transformation to be performed upon the dataset.
        :type transformation: Union[Callable[[Dict], Dict], Callable[[List[Dict]], List[Dict]]

        :param result_data_name: The name of the resulting data. Do not include path in result_data_name.
        :type result_data_name: str

        :param forced_rewrite: Whether to forcefully rewrite the existing file, if there is one.
        :type forced_rewrite: bool = False

        :param max_batch_size: If max_batch_size is specified and is >1, the transformation function must take inputs of type List[Dict] and return a List[Dict].
        :type max_batch_size: int = 1

        :param keep_key_fields: If keep_key_fields is True, the registered key_fields names will be copied to the new Data instance. Only do so if the transformation doesn't rename the key fields.
        :type keep_key_fields: bool = True

        :return: The data after transformation.
        :rtype: Data.
        """
        out_path = f"{root}/output/datasets/{result_data_name}.json"
        if self.data_name == result_data_name or self.data_path == out_path:
            warnings.warn(
                f"Data name {result_data_name} is the same as the current data name. The old instance will be invalidated."
            )
            return self.copy("temp_transform_artifact").transform(
                transformation,
                result_data_name,
                forced_rewrite,
                max_batch_size,
                keep_key_fields,
                map_key_fields,
            )
            
        Data.ask_and_remove_if_exists(out_path, forced_rewrite)

        def write_dict(sample_dict: Dict):
            nonlocal is_first, out_file
            out_file.write("\n" if is_first else ",\n")
            is_first = False
            out_file.write(json.dumps(sample_dict))
            # out_file.flush()
        
        def map_key_fields_fn(sample_dict: Dict) -> Dict:
            nonlocal self
            for k, v in self.default_key_fields.items():
                if k in self.key_fields and self.key_fields.get(k, v) != v and self.key_fields[k] in sample_dict:
                    sample_dict[v] = sample_dict[self.key_fields[k]]
                    del sample_dict[self.key_fields[k]]
            
            return sample_dict
        
        def inv_map_key_fields_fn(sample_dict: Dict) -> Dict:
            nonlocal self
            for k, v in self.default_key_fields.items():
                if v in sample_dict and self.key_fields.get(k, v) != v:
                    sample_dict[self.key_fields[k]] = sample_dict[v]
                    del sample_dict[v]
            
            return sample_dict

        with open(out_path, "w") as out_file:
            out_file.write("[")
            is_first = True

            if max_batch_size == 1:
                with tu.JsonListReader(self.data_path) as reader:
                    for element in reader:
                        if map_key_fields:
                            element = map_key_fields_fn(element)
                        
                        transformed = transformation(element)
                        if transformed is not None:
                            write_dict(transformed if not map_key_fields else inv_map_key_fields_fn(transformed))

            else:
                buffer = []

                with tu.JsonListReader(self.data_path) as reader:
                    for element in reader:
                        if map_key_fields:
                            element = map_key_fields_fn(element)
                        
                        buffer.append(element)
                        if len(buffer) == max_batch_size:
                            for e in transformation(buffer):
                                write_dict(e if not map_key_fields else inv_map_key_fields_fn(e))
                            buffer = []
                            out_file.flush()

                if buffer:
                    for e in transformation(buffer):
                        write_dict(e if not map_key_fields else inv_map_key_fields_fn(e))

            out_file.write("\n]")

        result = Data(result_data_name, self.data_type, out_path)
        if keep_key_fields:
            result.key_fields = self.key_fields.copy()
        return result

    def move_current_to_history(self):
        """
        Move the current dialogue turn in the prompt/question field and the response/predict field to the history field.
        
        :return: The data after the operation.
        :rtype: Data.
        """
        def move_to_history_fn(sample_dict: Dict) -> Dict:
            if sample_dict.get("instruction", "") or sample_dict.get("input", "") or sample_dict.get("output", "") or sample_dict.get("predict", ""):
                assert (sample_dict.get("instruction", "") or sample_dict.get("input", "")) and (sample_dict.get("output", "") or sample_dict.get("predict", ""))
                sample_dict["history"] = sample_dict.get("history", []) + [
                    [
                        sample_dict.get("instruction", "") + 
                            ("\n\n" if "instruction" in sample_dict and "input" in sample_dict else "") +
                            sample_dict.get("input", ""),
                        sample_dict.get("output", "") + sample_dict.get("predict", "")
                    ]
                ]
                sample_dict.pop("instruction", None)
                sample_dict.pop("input", None)
                sample_dict.pop("output", None)
                sample_dict.pop("predict", None)
            
            return sample_dict
        
        return self.transform(move_to_history_fn, self.data_name, forced_rewrite=True, map_key_fields=True)
    
    def switch_role_to_user(self, user_system_prompt: str = None, dialogue_starter: str = None):
        """
        Switch the prompt/question field and the response/predict field, thereby shifting the dialogue turn from the assistant to the user.
        
        :param user_system_prompt: The system prompt of the user role.
        :type user_system_prompt: str = None
        
        :param dialogue_starter: Placeholder message for the "zeroth" dialogue turn by the assistant that prompts the user to start the conversation.
        :type dialogue_starter: str = None
        
        :return: The data after the operation.
        :rtype: Data.
        """
        if user_system_prompt is None:
            user_system_prompt = "You are an assistant tasked with questioning the user, aka your partner. Ask informed questions to guide the conversation, follow up on the user's responses, and generally follow a natural conversation flow. Don't be too courteous; be concise."
        
        if dialogue_starter is None:
            dialogue_starter = "I am your partner. Please directly ask your first question."
        
        moved_to_history = self.move_current_to_history()
        
        def switch_role_to_user_fn(sample_dict: Dict) -> Dict:
            assert not (sample_dict.get("instruction", "") or sample_dict.get("input", "") or sample_dict.get("output", "") or sample_dict.get("predict", ""))
            
            all_histories = [h[i] for h in sample_dict.get("history", []) for i in range(2)]
            all_histories = [dialogue_starter] + all_histories
            assert len(all_histories) % 2 == 1
            sample_dict["history"] = [[all_histories[i], all_histories[i + 1]] for i in range(len(all_histories)-1, 2)]
            sample_dict["instruction"] = all_histories[-1]
            sample_dict["system"] = user_system_prompt
            return sample_dict
        
        return moved_to_history.transform(switch_role_to_user_fn, self.data_name, forced_rewrite=True, map_key_fields=True)
    
    def switch_role_to_assistant(self, assistant_system_prompt: str = None):
        """
        Switch the prompt/question field and the response/predict field, thereby shifting the dialogue turn from the user to the assistant.
        
        :param assistant_system_prompt: The system prompt of the assistant role.
        :type assistant_system_prompt: str = None
        
        :return: The data after the operation.
        :rtype: Data.
        """
        if assistant_system_prompt is None:
            assistant_system_prompt = "Please answer the user's questions. Be concise and not overly courteous, but be informative and provide all necessary details."
        
        moved_to_history = self.move_current_to_history()
        
        def switch_role_to_assistant_fn(sample_dict: Dict) -> Dict:
            assert not (sample_dict.get("instruction", "") or sample_dict.get("input", "") or sample_dict.get("output", "") or sample_dict.get("predict", ""))
            
            all_histories = [h[i] for h in sample_dict.get("history", []) for i in range(2)]
            assert len(all_histories) % 2 == 0
            sample_dict["history"] = [[all_histories[i], all_histories[i + 1]] for i in range(1, len(all_histories)-1, 2)]
            sample_dict["instruction"] = all_histories[-1]
            sample_dict["system"] = assistant_system_prompt
            return sample_dict

        return moved_to_history.transform(switch_role_to_assistant_fn, self.data_name, forced_rewrite=True, map_key_fields=True)
    
    def manage_llama_factory_registration(
        self, operation: Literal["add", "remove", "query"], forced_update: bool = True
    ) -> bool:
        """
        Add, remove, or query the registration status of the current dataset in Llama-Factory.
        No changes are made when adding an already existing dataset, or when removing a non-existent dataset.

        :param operation: The operation to perform. It can be "add", "remove", or "query".
        :type operation: Literal["add", "remove", "query"]

        :param forced_update: Whether to forcefully update the data
        :type forced_update: bool = True

        :return: A boolean meaning the registration status before this operation.
        :rtype: bool.
        """
        with open(f"{root}/libs/llama_factory/data/dataset_info.json", "r") as in_file:
            registrations = json.load(in_file)

        return_val = self.data_name in registrations

        if operation == "add":
            path = f"{root}/libs/llama_factory/data/{self.data_name}.json"
            if "llama_factory/data" not in self.data_path:
                Data.ask_and_remove_if_exists(path, forced_rewrite=True)
                os.system(f"cp {escape(self.data_path)} {escape(path)}")

        if operation == "add" and (forced_update or not return_val):
            path = f"{root}/libs/llama_factory/data/{self.data_name}.json"

            if "llama_factory" not in self.data_path:  # if is not built-in dataset
                if ("prompt" not in self.key_fields) or (
                    self.data_type == "sft" and "response" not in self.key_fields
                ):
                    warnings.warn(
                        "Please call set_key_fields() to specify the json fields where data is drawn from."
                    )
                elif self.data_type == "sft" and "input" not in self.key_fields:
                    warnings.warn(
                        "Suggestion: This is an instruction/SFT dataset. Do you want to use set_key_fields(query_field_name=...) to set input data, instead of putting input data into prompt?"
                    )

            registrations[self.data_name] = {
                "file_name": f"{self.data_name}.json",
                "columns": self.key_fields,
            }

            if self.data_type == "preference":
                registrations[self.data_name]["ranking"] = True

            print(
                f"Adding registration of data {self.data_name}: {registrations[self.data_name]}."
            )

            with open(f"{root}/libs/llama_factory/data/dataset_info.json", "w") as out_file:
                json.dump(registrations, out_file)

            print(f"Successfully completed registration of data {self.data_name}.")

        elif operation == "remove" and return_val:
            with open(
                f"{root}/libs/llama_factory/data/dataset_info_original.json", "r"
            ) as in_file:
                registrations_original = json.load(in_file)

            assert self.data_name not in registrations_original
            path = f'{root}/libs/llama_factory/data/{registrations[self.data_name]["file_name"]}'
            del registrations[self.data_name]

            with open(f"{root}/libs/llama_factory/data/dataset_info.json", "w") as out_file:
                json.dump(registrations, out_file)

            if os.path.exists(path):
                os.system(f"rm {escape(path)}")

            print(f"Successfully completed de-registration of data {self.data_name}.")

        else:
            print("No actions taken.")

        return return_val

    def set_key_fields(
        self,
        prompt_field_name: Optional[str] = None,
        query_field_name: Optional[str] = None,
        response_field_name: Optional[str] = None,
        system_field_name: Optional[str] = None,
        history_field_name: Optional[str] = None,
        suppress_registration_update: bool = False,
        **kwargs,
    ) -> None:
        """
        Specify which of the dict fields to use for training. In-place.

        Pass empty string to an argument in order to erase that argument.

        Will automatically update registration, if already registered.

        :param prompt_field_name: The name of the prompt field
        :type prompt_field_name: Optional[str] = None

        :param query_field_name: The name of the query field
        :type query_field_name: Optional[str] = None

        :param response_field_name: The name of the response field
        :type response_field_name: Optional[str] = None

        :param system_field_name: The name of the system field
        :type system_field_name: Optional[str] = None
        
        :param history_field_name: The name of the history field
        :type history_field_name: Optional[str] = None

        :param suppress_registration_update: Whether to suppress the update of the registration
        :type suppress_registration_update: bool = False

        Example:
            .. code-block:: python

                data.set_key_fields(prompt_field_name='content') # for pretraining dataset stored in content field
                data.set_key_fields(prompt_field_name='instruction', query_field_name='input', response_field_name='output') # for QA dataset with system prompt

        """
        if not suppress_registration_update:
            original_registration_status = self.manage_llama_factory_registration(
                "remove"
            )

        if prompt_field_name == "" and "prompt" in self.key_fields:
            del self.key_fields["prompt"]
        elif prompt_field_name:
            self.key_fields["prompt"] = prompt_field_name

        if query_field_name == "" and "query" in self.key_fields:
            del self.key_fields["query"]
        elif query_field_name:
            self.key_fields["query"] = query_field_name

        if response_field_name == "" and "response" in self.key_fields:
            del self.key_fields["response"]
        elif response_field_name:
            self.key_fields["response"] = response_field_name

        if system_field_name == "" and "system" in self.key_fields:
            del self.key_fields["system"]
        elif system_field_name:
            self.key_fields["system"] = system_field_name
        
        if history_field_name == "" and "history" in self.key_fields:
            del self.key_fields["history"]
        elif history_field_name:
            self.key_fields["history"] = history_field_name

        if isinstance(kwargs, dict):
            for k, v in kwargs.items():
                k_name = k.split("_field_name")[0]
                if v == "" and k_name in self.key_fields:
                    del self.key_fields[k_name]
                elif v:
                    self.key_fields[k_name] = v

        if not suppress_registration_update and original_registration_status:
            self.manage_llama_factory_registration("add")

    def save_permanent_and_register(
        self, saved_name: Optional[str] = None, forced_rewrite: bool = False
    ):
        """
        Data will be saved to :code:`data_save_path` from :code:`abstractions_config.json`.
        Without save_permanent, it will still be present in :code:`./output/` and can still be directly used next time without specifying the full path.
        Do not include path and suffix in the :code:`saved_name` argument.
        """
        saved_name = (saved_name or self.data_name).strip()
        if self.data_name not in saved_name.replace("/", ".").split("."):
            warnings.warn(
                f"Saved name {saved_name} doesn't match with data_name {self.data_name}"
            )

        if ".json" not in saved_name:
            saved_name += ".json"

        if "/" not in saved_name:
            path = os.path.join(data_save_path, saved_name)
        else:
            path = saved_name

        # if the path already exists, ask for approval before continuing
        Data.ask_and_remove_if_exists(path, forced_rewrite)

        # copy from data_path to path
        execute(f"cp {escape(self.data_path)} {escape(path)}")
        print(f"Successfully saved to {path}.")

        self.manage_llama_factory_registration("add")

    def all_passages(self) -> Iterable[Dict[Hashable, Any]]:
        """
        Returns an iterator of all passages (json dicts) in this dataset.
        """
        with tu.JsonListReader(self.data_path) as reader:
            for element in reader:
                yield element


class DataFileCollection:
    name2collection: Dict[str, Any] = {}
    always_force_rewrite: bool = True
    """
    The Data File Collection class stores multi-file data, by name, path and type, etc. Before being used for training, the Data File Collection needs to be converted to Data.
    Operations similar to those of Data are available in this class nevertheless.
    """

    # check with user before removing a file
    @classmethod
    def ask_and_remove_if_exists(cls, path: str, forced_rewrite: bool):
        if os.path.exists(path):
            if forced_rewrite or (
                hasattr(cls, "always_force_rewrite") and cls.always_force_rewrite
            ):
                execute(f'rm {"-r" if os.path.isdir(path) else ""} -f {escape(path)}')
                return

            warnings.warn(
                f"{path} already exists. Use forced_rewrite=True to force rewrite."
            )
            answer = input("Do you want to force rewrite? (yes/no/always) ").lower()
            if "n" in answer:
                return
            if "a" in answer:
                cls.always_force_rewrite = True
            execute(f'rm {"-r" if os.path.isdir(path) else ""} {escape(path)}')

    def __init__(
        self,
        collection_name: str,
        data_type: Literal["pretrain", "sft", "preference"] = "pretrain",
        collection_path: Optional[str] = None,
        file_selection_func: Optional[Callable[[str], bool]] = None,
        **kwargs,
    ):
        """
        Initialize.

        :param prompt_field_name: The name of the prompt field
        :type prompt_field_name: Optional[str] = None

        :param query_field_name: The name of the query field
        :type query_field_name: Optional[str] = None

        :param response_field_name: The name of the response field
        :type response_field_name: Optional[str] = None

        :param system_field_name: The name of the system field
        :type system_field_name: Optional[str] = None

        :param suppress_registration_update: Whether to suppress the update of the registration
        :type suppress_registration_update: bool = False

        If collection_path is omitted, we will search for collection_name in directories specified in abstractions_config.json.
        When file_selection_func is supplied, files will be captured real-time, instead of only when initializing. Only json files will be captured.
        You may want to exclude undated.json using file_selection_func. That file is huge.

        Example:
            .. code-block:: python

                DataFileCollection(collection_name='histtext_1826_to_2018',
                                data_type='pretrain',
                                collection_path = f'{root}/dataset/dataset_text_sequence/',
                                file_selection_func = (lambda path: 1826 <= int(path.split('/')[-1][1:6]) <= 2018))

        """
        # if collection_name in DataFileCollection.name2collection:
        #     warnings.warn(f'The collection name {collection_name} is already in use.')

        if "is_instruction_data" in kwargs:
            warnings.warn(
                f"is_instruction_data is deprecated. Please use data_type instead."
            )
            data_type = "sft" if kwargs["is_instruction_data"] else "pretrain"

        self.collection_name = collection_name
        self.collection_path = collection_path
        self.data_type = data_type
        self.file_selection_func = file_selection_func

        # if collection_path is not specified, look for it in the paths specified in abstractions_config.json
        if not collection_path:
            for search_path in data_search_paths:
                if os.path.exists(os.path.join(search_path, collection_name)):
                    print(
                        f"Found collection {collection_name} at {os.path.join(search_path, collection_name)}"
                    )
                    self.collection_path = os.path.join(search_path, collection_name)
                    break

        if not self.collection_path or not os.path.exists(self.collection_path):
            raise FileNotFoundError(
                f"The collection path {self.collection_path} doesn't exist."
            )
        else:
            self.collection_path = self.collection_path.rstrip("/")

        if self.collection_name in DataFileCollection.name2collection:
            DataFileCollection.name2collection[self.collection_name].append(self)
        else:
            DataFileCollection.name2collection[self.collection_name] = [self]

    def copy(self) -> "DataFileCollection":
        """Returns a shallow copy of the current DataFileCollection instance."""
        return DataFileCollection(
            self.collection_name,
            self.data_type,
            self.collection_path,
            self.file_selection_func,
        )

    def all_files(self) -> Iterable[str]:
        """
        Returns an iterator of all json files in this collection.
        If file_selection_func had been specified, files will be captured real-time, instead of only when initializing.
        """
        for root, dirs, files in os.walk(self.collection_path):
            for file in files:
                path = os.path.join(root, file).strip()
                if path.endswith(".json") and (
                    not self.file_selection_func or self.file_selection_func(path)
                ):
                    yield path

    def all_passages(self) -> Iterable[Dict[Hashable, Any]]:
        """
        Returns an iterator of all passages (json dicts) in this collection.
        If file_selection_func had been specified, files will be captured real-time, instead of only when initializing.
        """
        for in_path in tqdm(
            list(self.all_files())
        ):  # remove list() if it turns out that the file count is super huge
            assert in_path[: len(self.collection_path)] == self.collection_path
            with tu.JsonListReader(in_path) as reader:
                for element in reader:
                    yield element

    def transform(
        self,
        transformation: Union[
            Callable[[Dict], Optional[Dict]], Callable[[List[Dict]], List[Dict]]
        ],
        result_collection_name: str,
        forced_rewrite: bool = False,
        max_batch_size: int = 1,
        suppress_tqdm: bool = False,
    ):
        """
        Apply transformation to every element of the current dataset (in the format of a json list of json dicts where the values are of mutable or immutable types), and returns a DataFileCollection instance containing the resulting dataset.
        Out-of-place. Does not modify self.

        This function (like all others in abstractions) is memory-efficient for huge json files.
        All data files should be json files with the type of List[Dict[Hashable, Any]].

        :param transformation: Transformation applied to every element of the current dataset
        :type transformation: Union[Callable[[Dict], Optional[Dict]], Callable[[List[Dict]], List[Dict]]]

        :param result_collection_name: The name of the resulting collection. Do not include path in result_collection_name.
        :type result_collection_name: str

        :param forced_rewrite: Whether to forcefully rewrite the existing data
        :type forced_rewrite: bool = False

        :param max_batch_size: The maximum batch size. If max_batch_size is specified and is >1, the transformation function must take inputs of type List[Dict] and return a List[Dict].
        :type max_batch_size: int = 1

        :param suppress_tqdm: Whether to suppress the tqdm progress bar
        :type suppress_tqdm: bool = False
        """
        result_dir = f"{root}/output/datasets/{result_collection_name}/"
        DataFileCollection.ask_and_remove_if_exists(result_dir, forced_rewrite)
        os.makedirs(result_dir, exist_ok=True)

        def write_dict(sample_dict: Dict):
            nonlocal is_first, out_file
            out_file.write("\n" if is_first else ",\n")
            is_first = False
            out_file.write(json.dumps(sample_dict))
            # out_file.flush()

        for in_path in (
            self.all_files() if suppress_tqdm else tqdm(list(self.all_files()))
        ):  # remove list() if it turns out that the file count is super huge
            assert in_path[: len(self.collection_path)] == self.collection_path
            out_path = os.path.join(
                result_dir, in_path.replace(self.collection_path, "").lstrip("/")
            )
            out_path_dir = "/".join(out_path.split("/")[:-1])
            if not os.path.isdir(out_path_dir):
                os.makedirs(out_path_dir)

            with open(out_path, "w") as out_file:
                out_file.write("[")
                is_first = True

                if max_batch_size == 1:
                    with tu.JsonListReader(in_path) as reader:
                        for element in reader:
                            transformed = transformation(element)
                            if transformed is not None:
                                write_dict(transformed)

                else:
                    buffer = []

                    with tu.JsonListReader(in_path) as reader:
                        for element in reader:
                            buffer.append(element)
                            if len(buffer) == max_batch_size:
                                for e in transformation(buffer):
                                    write_dict(e)
                                buffer = []
                                out_file.flush()

                    if buffer:
                        for e in transformation(buffer):
                            write_dict(e)

                out_file.write("\n]")

        return DataFileCollection(result_collection_name, self.data_type, result_dir)

    def save_permanent(
        self, saved_name: Optional[str] = None, forced_rewrite: bool = False
    ):
        """
        DataFileCollection will be saved to :code:`data_save_path` from :code:`abstractions_config.json`.
        Without save_permanent, it will still be present in :code:`./output/` and can still be directly used next time without specifying the full path.
        Normally, you should not include full path and/or suffix in :code:`saved_name`. If you do, it will be seen as a path. In this case, the collection may not be autodiscovered by abstractions for future use.
        """
        saved_name = (saved_name or self.collection_name).strip()
        if self.collection_name not in saved_name.replace("/", ".").split("."):
            warnings.warn(
                f"Saved name {saved_name} doesn't match with collection_name {self.collection_name}"
            )

        if "/" not in saved_name:
            path = os.path.join(data_save_path, saved_name)
        else:
            path = saved_name

        # if the path already exists, ask for approval before continuing
        DataFileCollection.ask_and_remove_if_exists(path, forced_rewrite)

        # copy from collection_path to path
        execute(f"cp -r {escape(self.collection_path)} {escape(path)}")
        print(f"Successfully saved to {path}.")

    def convert_to_Data(
        self, result_data_name: str, forced_rewrite: bool = False, filter_fields=None
    ):
        """
        Convert self to an Data instance by merging all data files into one json, and return that Data instance with name result_data_name.
        Out-of-place. Does not modify self, and self is still usable after this operation.
        All data files should be json files with the type of List[Dict[Hashable, Any]].

        :param result_data_name: The name of the resulting data
        :type result_data_name: str

        :param forced_rewrite: Whether to forcefully rewrite the existing data
        :type forced_rewrite: bool = False

        :param filter_fields: Fields to filter the data (default is None)
        :type filter_fields: Optional = None
        """
        path = f"{root}/output/datasets/{result_data_name}.json"
        Data.ask_and_remove_if_exists(path, forced_rewrite)

        with open(path, "w") as out_file:
            out_file.write("[")
            is_first = True

            for in_path in tqdm(
                list(self.all_files())
            ):  # remove list() if it turns out that the file count is super huge
                with tu.JsonListReader(in_path) as reader:
                    for element in reader:
                        out_file.write("\n" if is_first else ",\n")
                        is_first = False
                        out_file.write(json.dumps(clean_dict(element, filter_fields)))

            out_file.write("\n]")

        return Data(result_data_name, self.data_type, path)

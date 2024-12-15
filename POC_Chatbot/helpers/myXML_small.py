import os
from llama_index.core import PromptTemplate
import json
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from typing import List
from llama_index.core.postprocessor import SentenceTransformerRerank

# Functions used in the streamlit

def get_id_lookup_dict_for_final_title_print(base_path_MANUALS):
    lookup_dict = {}
    my_path = base_path_MANUALS + "/storage/dict_title_filename_document"

    for filename in os.listdir(my_path):
        with open(os.path.join(my_path, filename), 'r') as f:
            data = json.load(f)
        if len(set(lookup_dict.keys()).intersection(set(data.keys()))) > 0:
            assert False, "keys should be unqiue"
        else:
            lookup_dict.update(data)
    return lookup_dict


def dict_to_markdown_table(data):
    columns = list(data.keys())
    rows = list(zip(*data.values()))

    # Construct the Markdown table
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows_md = "\n".join("| " + " | ".join(row) + " |" for row in rows)

    return f"{header}\n{separator}\n{rows_md}"


def load_indeces_retriever(indeces_topk: dict) -> list:
    option = "_llama3B_bgeL"
    res = []
    for i, k in indeces_topk:
        storage_context = StorageContext.from_defaults(
            persist_dir="./_MANUALS/storage/" + i + option
        )
        index = load_index_from_storage(storage_context)
        res.append(VectorIndexRetriever(index=index, similarity_top_k=k))
    return res


def my_retriever_builder(indices_topk, llm, reranker=None):
    indices = load_indeces_retriever(indices_topk)
    custom_retriever = CustomRetriever(indices, llm, reranker)
    return custom_retriever


class CustomRetriever(BaseRetriever):
    """Looks up the query in all indices and concatenates the retrieved nodes to one list."""

    def __init__(
            self,
            vector_retrievers: List[VectorIndexRetriever],
            llm,
            my_reranker: SentenceTransformerRerank
    ) -> None:
        """Init params."""
        self._vector_retrievers = vector_retrievers
        self.my_reranker = my_reranker
        self.llm = llm
        self.applicable_index_retrieval_template = (
            "The prompt you will receive is a question,that should be aircraft related. In the prompt, there are most "
            "likely gonna be aircraft models included. Typical aircraft models include:"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Please identify if the question is to any extend aircraft related. If this is the case, return 'True', "
            "if not return 'False'. It is important that you respond with either True or False and really only return "
            "True, if is actually related to the query."
            "The query is: {query_str}. Explain your answer please.\n"
        )
        self.contex_aircraft_models = 'A320, A321, A322, A380, Airbus, Boeing'
        super().__init__()

    def my_rerank(self, vector_nodes, query_bundle):
        return self.my_reranker.postprocess_nodes(nodes=vector_nodes, query_bundle=query_bundle)

    def find_applicable_index(self, query_bundle):
        prompt_template = PromptTemplate(
            self.applicable_index_retrieval_template)
        message = prompt_template.format_messages(context_str=self.contex_aircraft_models,
                                                  query_str=query_bundle.query_str)

        response = self.llm.chat(message)

        if response is None or 'True' in response.message.content:
            print("Related to the question")
            return self._vector_retrievers
        else:
            print("Not related to the question")
            return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        print(f"Retrieval for following query: {query_bundle.query_str}")

        applic_idx = self.find_applicable_index(query_bundle)

        vector_nodes = []
        for idx in applic_idx:
            print("Retrieval on index-------------")
            vector_nodes += idx.retrieve(query_bundle)

        if self.my_reranker:
            print(
                f"BEFORE reranking we have following {len(vector_nodes)} retrieved nodes -----------------------")
            print(vector_nodes)

            vector_nodes = self.my_rerank(vector_nodes, query_bundle)

            print(
                f"AFTER reranking we have following {len(vector_nodes)} retrieved nodes -----------------------")
            print(vector_nodes)

        # assert False, "remove this if you want the LLM to get queried with the vector nodes to draft an answer"

        return vector_nodes

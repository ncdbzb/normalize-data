from dotenv import load_dotenv
import os
from operator import itemgetter
from textwrap import dedent

from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.retrievers.yandex_search import YandexSearchAPIRetriever
from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
load_dotenv()
from parsing_xlsx import parse_xlsx
from classification import get_classes_v1
from yandex_search import get_search_chain
from normalize_data import get_normalize_chain

def main():

    # YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
    # YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
    # GIGACHAT_CREDENTIALS = os.getenv('GIGACHAT_CREDENTIALS')

    giga = GigaChat(
        # credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
    )
    # Парсинг данных из xlsx
    parsed_data = parse_xlsx('normalize_data.xlsx')

    # Создание классов
    classes_v1 = get_classes_v1(parsed_data)
    # first_class_demo = list(classes_v1.keys())[3]
    first_class_demo = 'Автошина'
    print(f'FIRST CLASS: {first_class_demo}\n')
    # first_class_items_list = list(classes_v1.values())[0]
    first_class_items_list = ['Автошина LINGLONG 315/80R22.5 D960 LEAO']
    print(f'RECORDS FOR FIRST CLASS: {first_class_items_list}\n\n')

    # Создание шаблона для класса
    search_chain = get_search_chain(giga)
    response = search_chain.invoke({"question": first_class_demo})
    first_class_template = response["answer"]
    print(f'FIRST CLASS TEMPLATE: {first_class_template}')
    tokens = giga.tokens_count([response["answer"], response["question"], ' '.join([doc.page_content for doc in response["context"]])])
    sources = "\n\n".join(
        f'{doc.page_content} ({doc.metadata["url"]})' for doc in response["context"]
    )
    print(sources)
    print(f'SPENT TOKENS: {tokens}\n\n')


    # Нормализация данных по шаблону
    normalize_chain = get_normalize_chain(giga)
    # Записи, относящие к первому классу
    for record in first_class_items_list:
        result = normalize_chain.invoke({"template": first_class_template, "product_name": record})
        print(f'RECORD: {record}')
        print(f'NORMALIZED DATA: {result.content}')
        print(f'SPENT TOKENS: {result.response_metadata["token_usage"].total_tokens}')


if __name__ == '__main__':
    # query = "Автохолодильник"
    # response = chain_with_source.invoke({"question": query})
    # print(response["answer"])

    # result = chain.invoke({"template": "[Тип холодильника] [Температура охлаждения] [Мощность] [Объем] [Материал корпуса] [Внешний вид] [Функции]", "product_name": "Автохолодильник Apricool C40"})
    # print(result.content)
    # parsed_data = parse_xlsx('normalize_data.xlsx')
    # classes_v1 = get_classes_v1(parsed_data)
    main()



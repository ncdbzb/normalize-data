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

from parsing import parse_xlsx


load_dotenv()

# YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
# YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
GIGACHAT_CREDENTIALS = os.getenv('GIGACHAT_CREDENTIALS')


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# def generate_final_answer(response):
#     final_answer_template = dedent(
#         """
#         ### Вопрос
#         {question}
#
#         ### Ответ
#         {answer}
#
#         ### Источники
#         {sources}"""
#     ).strip()
#
#     final_answer = final_answer_template.format(
#         question=response["question"],
#         answer=response["answer"],
#         sources="\n\n".join(
#             f'{doc.page_content} ({doc.metadata["url"]})' for doc in response["context"]
#         ),
#     )
#
#     return final_answer


model = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    verify_ssl_certs=False,
)

api_wrapper = YandexSearchAPIWrapper()

retriever = YandexSearchAPIRetriever(api_wrapper=api_wrapper, k=5)


QA_TEMPLATE = """Ты — ИИ, который помогает выявлять атрибуты для различных классов товаров. У меня есть класс товаров, для которого нужно определить точный набор атрибутов, описывающих его структуру. Используй данные из переменной "context", в которой содержится информация с нескольких релевантных сайтов.

### Задача:
Анализируй "context", чтобы определить наиболее точный набор атрибутов, который описывает структуру товаров для данного класса. Пример класса: "Гвоздь".

### Пример:
Для класса "Гвоздь" правильный шаблон может выглядеть следующим образом:
[Вид продукции] [Тип гвоздя] [Диаметр гвоздя, мм] х [Длина гвоздя, мм] [Покрытие] [Стандарт]

Используй этот пример как ориентир для выявления структуры атрибутов класса товаров, которые нужно нормализовать. Шаблон должен быть максимально универсальным для всех товаров данного класса.

### Требования:
- Проведи исследование релевантных источников, чтобы найти общепринятые атрибуты для данного класса товаров.
- Верни шаблон с указанием всех возможных атрибутов в правильном порядке (например, диаметр, длина, покрытие).
- Шаблон должен учитывать вариации описания товаров в пределах данного класса (например, опциональные атрибуты, такие как покрытие или стандарт).
- Убедись, что предложенный шаблон применим для нормализации всех товаров, относящихся к данному классу.
- В ответе не пиши ничего лишнего, нужен только шаблон в правильном формате

### Входные данные:
Класс товаров: "{question}"

### Выходные данные:
Структурированный шаблон для товаров данного класса с указанием всех возможных атрибутов в правильном порядке.

### Context:
{context}
"""
prompt = ChatPromptTemplate.from_template(QA_TEMPLATE)


output_parser = StrOutputParser()

chain_without_source = (
        RunnableParallel(
            {
                "context": itemgetter("context") | RunnableLambda(format_docs),
                "question": itemgetter("question"),
            }
        )
        | prompt
        | model
        | output_parser
)
chain_with_source = RunnableParallel(
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
).assign(answer=chain_without_source)

normalize_prompt = """
Ты — ИИ, который помогает нормализовать номенклатуру на основе заданного шаблона. У меня есть неструктурированная запись номенклатуры, и мне нужно, чтобы ты разнёс её по атрибутам согласно заданному шаблону.

### Задача:
Ты получишь:
1. Шаблон для данного класса товаров с указанием всех атрибутов в структурированном виде.
2. Неструктурированную запись номенклатуры, которую нужно нормализовать.

Твоя задача — правильно распределить элементы неструктурированной записи по соответствующим атрибутам в шаблоне. Если каких-либо данных не хватает или они отсутствуют в записи, оставь это поле пустым.

### Пример:
- Неструктурированная запись: "Гвоздь медный 3,5х35мм"
- Шаблон:
"[Вид продукции] [Тип гвоздя] [Диаметр гвоздя, мм] х [Длина гвоздя, мм] [Покрытие] [Стандарт]"

### Входные данные:
1. Шаблон: {template}
2. Неструктурированная запись: "{product_name}"

### Выходные данные:
Словарь, где ключами являются атрибуты из шаблона, а значениями — соответствующие данные из записи. Если какое-либо значение отсутствует, ключ должен содержать пустую строку.

### Требования:
- Используй структуру шаблона для корректного распределения атрибутов.
- Верни данные в формате словаря, где ключи — это атрибуты из шаблона, а значения — соответствующие элементы из номенклатуры.
- Убедись, что нормализованные данные точны, и если информация отсутствует, соответствующие поля должны оставаться пустыми.
"""

normalize_prompt = ChatPromptTemplate.from_template(normalize_prompt)

chain = normalize_prompt | model

if __name__ == '__main__':
    # query = "Автохолодильник"
    # response = chain_with_source.invoke({"question": query})
    # print(response["answer"])

    result = chain.invoke({"template": "[Тип холодильника] [Температура охлаждения] [Мощность] [Объем] [Материал корпуса] [Внешний вид] [Функции]", "product_name": "Автохолодильник Apricool C40"})
    print(result.content)

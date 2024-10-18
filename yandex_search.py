from operator import itemgetter

from langchain_community.retrievers.yandex_search import YandexSearchAPIRetriever
from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

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

def get_search_chain(model):
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

    return chain_with_source
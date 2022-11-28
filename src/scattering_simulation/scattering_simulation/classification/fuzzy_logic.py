from abc import ABC, abstractmethod
from .membership_functions import *


class BaseClassifier(ABC):
    def __init__(self, classes: list, products: list):
        self.classes = classes
        self.products = products

    @abstractmethod
    def classify(self, products: dict) -> str:
        pass


class FuzzyClassifier(BaseClassifier):
    def __init__(self):
        self.membership_funcs = get_membership_functions()
        super().__init__(get_classes_list(), get_products_list())

    def classify(self, products_in: dict) -> str:
        if not products_in.keys() == self.products:
            print(products_in.keys(), self.products)
            print(f'fuzzy_logic.FuzzyClassifier.classify: Unknown product keys received {list(products_in.keys())}.\n'
                  f'Expecting to receive keys {list(self.products)} Exiting!')
            quit()
        accumulator = {key: 0 for key in self.classes}
        for class_name in self.classes:
            for product_name in self.products:
                accumulator[class_name] += self.membership_funcs[class_name][product_name](products_in[product_name])
        return max(accumulator, key=accumulator.get)

    def classify_weighted(self, products_in: dict) -> str:

        if not products_in.keys() == self.products:
            print(products_in.keys(), self.products)
            print(f'fuzzy_logic.FuzzyClassifier.classify: Unknown product keys received {list(products_in.keys())}.\n'
                  f'Expecting to receive keys {list(self.products)} Exiting!')
            quit()
        accumulator = {key: 0 for key in self.classes}
        for class_name in self.classes:
            for product_name in self.products:
                accumulator[class_name] += get_weight(class_name, product_name) * \
                                           self.membership_funcs[class_name][product_name](products_in[product_name])
            accumulator[class_name] /= get_total_weight(class_name)

        return max(accumulator, key=accumulator.get)


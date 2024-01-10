import time
from CalculateObject import CalculateObject
from CalculationParameters import CalculationParameters

from PyQt5.QtCore import pyqtSignal, QObject


# Create calculate thread class
class NewCalculateThread(QObject):
    def __init__(self, window):
        super().__init__()
        self.calculation_was_canceled = 0
        self.window = window
        self.calculator: CalculateObject = CalculateObject()

    results_signal = pyqtSignal(list)

    def run(self):
        calculation_params: CalculationParameters = CalculationParameters()

        calculation_params.part_names = self.window.part_names
        calculation_params.part_quantities = self.window.part_quantities
        calculation_params.part_lengths = self.window.part_lengths
        calculation_params.max_containers = self.window.max_containers
        calculation_params.max_parts_per_nest = self.window.max_parts_per_nest
        calculation_params.stock_length = self.window.stock_length
        calculation_params.left_waste = self.window.left_waste
        calculation_params.right_waste = self.window.right_waste
        calculation_params.spacing = self.window.spacing
        calculation_params.error = self.window.error

        nesting_start_time = time.time()
        [final_patterns, final_allocations, warnings] = self.calculator.length_nest_pro_calculate(calculation_params)
        warnings.append(f"nesting time was {time.time() - nesting_start_time}")
        results = [final_patterns, final_allocations, warnings]

        # noinspection PyUnresolvedReferences
        self.results_signal.emit(results)


import time
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from window import Window
from NewWindow import NewWindow

# TODO add results outputs such as scrap %
# TODO number or letter each pattern
# TODO add scale to patterns display (possibly behind them), or add person or car to side for size reference
# TODO add explain on hover
# TODO add help documentation to explain nesting algorithms and how to use the application
# TODO add more colors and/or patterns for part display
# TODO allow user to click on a nest pattern to show it in more detail
# TODO implement a xml template for default settings that opens on launch

# Start timer
application_start_time = time.time()

# Create the application
app = QApplication(sys.argv)

# Force the style to be the same on all OSs:
app.setStyle("Fusion")

# Use a palette to switch to dark colors:
palette = QPalette()
palette.setColor(QPalette.Window, QColor(53, 53, 53))
palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
palette.setColor(QPalette.Base, QColor(25, 25, 25))
palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
palette.setColor(QPalette.Text, QColor(255, 255, 255))
palette.setColor(QPalette.Button, QColor(53, 53, 53))
palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
palette.setColor(QPalette.Link, QColor(42, 130, 218))
palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

app.setPalette(palette)


def main():
    # Create window
    window = NewWindow()
    window.app = app

    # Show window
    window.show()
    window.t1.update_table_width(window.t1)

    # Exit when finished
    sys.exit(app.exec_())


# Run Program
main()

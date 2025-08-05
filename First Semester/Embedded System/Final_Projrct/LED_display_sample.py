from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.virtual import viewport
from luma.core.legacy import text, show_message
from luma.core.legacy.font import (
	proportional, 
	CP437_FONT,  
	TINY_FONT, 
	SINCLAIR_FONT, 
	LCD_FONT) 

import time

#Create a serial instance and specify SPI bus parameters
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=0)
#cascaded=1 means only one device is connected, block_orientation=[0, 90, -90], Corrects block orientation when wired vertically.
#show message
time.sleep(1)
msg = "61375017H, 61375079H"
show_message(device, msg, fill="white", font=proportional(LCD_FONT), scroll_delay=0.1) #whitemeans LED is illuminated
time.sleep(1)
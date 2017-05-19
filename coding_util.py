import os

def checkAndCreate(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
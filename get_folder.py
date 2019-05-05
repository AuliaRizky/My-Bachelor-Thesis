

def get_folder_name(folder):
    folder_name = ['28', '30', '31', '32', '33', '35']
    return next(folder_name for folder_name in folder_name if folder_name in folder)
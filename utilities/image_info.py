class ImageInfo:
    def __init__(self, image, input_tensor):
        self.image = image
        self.input_tensor = input_tensor

        # Letterboxing info
        self.letterboxed = False
        self.letterbox_offset = 0

    def set_letterbox(offset):
        self.letterboxed = True
        self.letterbox_offset = offset

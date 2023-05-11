import numpy as np

class VideoCaptureYUV(object):
    def __init__(self, filename, resolution):
        self.file = open(filename, 'rb')
        self.width, self.height = resolution
        self.uv_width = self.width // 2
        self.uv_height = self.height // 2

    def read_frame(self):
        Y = self.read_channel(self.height, self.width)
        U = self.read_channel(self.uv_height, self.uv_width)
        V = self.read_channel(self.uv_height, self.uv_width)
        return Y, U, V

    def read_channel(self, height, width):
        channel_len = height * width
        shape = (height, width)

        raw = self.file.read(channel_len)
        channel = np.frombuffer(raw, dtype=np.uint8)
        
        channel = channel.reshape(shape)

        return channel

    def close(self):
        self.file.close()


# In[12]:


def calculate_mse(original, encoded):
    # Convert frames to double
    original = np.array(original, dtype=np.double)
    encoded = np.array(encoded, dtype=np.double)

    # Calculate mean squared error
    mse = np.mean((original - encoded) ** 2)
    return mse

# In[13]:


def calculate_psnr(original, encoded, resolution, frames):
    MAX_VALUE = 255
    original_video = VideoCaptureYUV(original, resolution)
    encoded_video = VideoCaptureYUV(encoded, resolution)

    mse_array = list()

    for frame in range(frames):
        original_y, _, _ = original_video.read_frame()
        encoded_y, _, _ = encoded_video.read_frame()

        mse_y = calculate_mse(original_y, encoded_y)
        mse_array.append(mse_y)

    # Close YUV streams
    original_video.close()
    encoded_video.close()

    # Calculate YUV-PSNR based on average MSE
    mse = np.average(mse_array)
    if mse == 0: psnr = MAX_VALUE
    else: psnr = 10 * np.log10((MAX_VALUE * MAX_VALUE) / mse)

    return psnr, mse_array






from __future__ import division
import math
import numpy as np
import cv2
# import and use one of 3 libraries PIL, cv2, or scipy in that order
USE_PIL = True
USE_CV2 = False
USE_SCIPY = False
try:
    import PIL
    from PIL import Image
    raise ImportError
except ImportError:
    USE_PIL = False
if not USE_PIL:
    USE_CV2 = True
    try:
        import cv2
    except ImportError:
        USE_CV2 = False
if not USE_PIL and not USE_CV2:
    USE_SCIPY = True
    try:
        import scipy
        from scipy import misc
    except ImportError:
        USE_SCIPY = False
        raise RuntimeError("couldn't load ANY image library")

class _OtsuPyramid(object):
    """segments histogram into pyramid of histograms, each histogram
    half the size of the previous. Also generate omega and mu values
    for each histogram in the pyramid.
    """

    def load_image(self, im, bins=256):
        """ bins is number of intensity levels """
        if not type(im) == np.ndarray:
            raise ValueError(
                'must be passed numpy array. Got ' + str(type(im)) +
                ' instead'
            )
        if im.ndim == 3:
            raise ValueError(
                'image must be greyscale (and single value per pixel)'
            )
        self.im = im
        hist, ranges = np.histogram(im, bins) #将输入转化为直方图
        # print("hist:",hist,"range:",ranges) # hist表示每个区间内的元素个数，ranges表示区间 
        # convert the numpy array to list of ints
        hist = [int(h) for h in hist]
        histPyr, omegaPyr, muPyr, ratioPyr = \
            self._create_histogram_and_stats_pyramids(hist)
        # arrange so that pyramid[0] is the smallest pyramid
        self.omegaPyramid = [omegas for omegas in reversed(omegaPyr)] # 前缀和
        #  print("self.omeaPtramid:",len(self.omegaPyramid[0]))
        self.muPyramid = [mus for mus in reversed(muPyr)]# 乘了像素的 前缀和
        self.ratioPyramid = ratioPyr# [1, 2, 2, 2, 2, 2, 2, 2]
        
    def _create_histogram_and_stats_pyramids(self, hist):
        """Expects hist to be a single list of numbers (no numpy array)
        takes an input histogram (with 256 bins) and iteratively
        compresses it by a factor of 2 until the last compressed
        histogram is of size 2. It stores all these generated histograms
        in a list-like pyramid structure. Finally, create corresponding
        omega and mu lists for each histogram and return the 3
        generated pyramids.
        """
        bins = len(hist)
        # eventually you can replace this with a list if you cannot evenly
        # compress a histogram
        ratio = 2
        reductions = int(math.log(bins, ratio)) #ln(bins)/ln(ratio)，约等于开方
        compressionFactor = []
        histPyramid = []
        omegaPyramid = []
        muPyramid = []
        for _ in range(reductions):
            histPyramid.append(hist)
            reducedHist = [sum(hist[i:i+ratio]) for i in range(0, bins, ratio)]
            # collapse a list to half its size, combining the two collpased
            # numbers into one
            hist = reducedHist
            # update bins to reflect the img_nums of the new histogram
            bins = bins // ratio
            compressionFactor.append(ratio)
        # first "compression" was 1, aka it's the original histogram
        compressionFactor[0] = 1
        # print("the length of histPyramid:",len(histPyramid))
        for hist in histPyramid:
            omegas, mus, muT = \
                self._calculate_omegas_and_mus_from_histogram(hist)
            omegaPyramid.append(omegas)
            muPyramid.append(mus)
        return histPyramid, omegaPyramid, muPyramid, compressionFactor

    def _calculate_omegas_and_mus_from_histogram(self, hist):
        """ Comput histogram statistical data: omega and mu for each
        intensity level in the histogram
        """
        probabilityLevels, meanLevels = \
            self._calculate_histogram_pixel_stats(hist)
        bins = len(probabilityLevels)
        # these numbers are critical towards calculations, so we make sure
        # they are float
        ptotal = float(0)
        # sum of probability levels up to k
        omegas = []
        for i in range(bins):
            ptotal += probabilityLevels[i]
            omegas.append(ptotal)
        mtotal = float(0)
        mus = []
        for i in range(bins):
            mtotal += meanLevels[i]
            mus.append(mtotal)
        # muT is the total mean levels.
        muT = float(mtotal)
        return omegas, mus, muT

    def _calculate_histogram_pixel_stats(self, hist):
        """Given a histogram, compute pixel probability and mean
        levels for each bin in the histogram. Pixel probability
        represents the likely-hood that a pixel's intensty resides in
        a specific bin. Pixel mean is the intensity-weighted pixel
        probability.
        """
        # bins = number of intensity levels
        bins = len(hist)
        # print("bins:",bins)
        # N = number of pixels in image. Make it float so that division by
        # N will be a float
        N = float(sum(hist))
        # percentage of pixels at each intensity level: i => P_i
        hist_probability = [hist[i] / N for i in range(bins)]
        # mean level of pixels at intensity level i   => i * P_i
        pixel_mean = [i * hist_probability[i] for i in range(bins)]
        # print("N:",N)
        return hist_probability, pixel_mean


class OtsuFastMultithreshold(_OtsuPyramid):
    """Sacrifices precision for speed. OtsuFastMultithreshold can dial
    in to the threshold but still has the possibility that its
    thresholds will not be the same as a naive-Otsu's method would give
    """

    def calculate_k_thresholds(self, k):
        self.threshPyramid = []
        start = self._get_smallest_fitting_pyramid(k)
        self.bins = len(self.omegaPyramid[start])
        # print("self.bins:",self.bins)
        thresholds = self._get_first_guess_thresholds(k)
        # print("thresholds:",thresholds)
        # give hunting algorithm full range so that initial thresholds
        # can become any value (0-bins)
        deviate = self.bins // 2
        for i in range(start, len(self.omegaPyramid)):
            omegas = self.omegaPyramid[i] # 个数平均前缀和
            mus = self.muPyramid[i] # 平均像素前缀和
            hunter = _ThresholdHunter(omegas, mus, deviate)
            thresholds = \
                hunter.find_best_thresholds_around_estimates(thresholds)
            self.threshPyramid.append(thresholds)
            # how much our "just analyzed" pyramid was compressed from the
            # previous one
            scaling = self.ratioPyramid[i]
            # deviate should be equal to the compression factor of the
            # previous histogram.
            deviate = scaling
            thresholds = [t * scaling for t in thresholds]
        # return readjusted threshold (since it was scaled up incorrectly in
        # last loop)
        # print("true thresholds:",thresholds)
        return [t // scaling for t in thresholds]

    def _get_smallest_fitting_pyramid(self, k):
        """Return the index for the smallest pyramid set that can fit
        K thresholds
        """
        for i, pyramid in enumerate(self.omegaPyramid):
            if len(pyramid) >= k:
                return i

    def _get_first_guess_thresholds(self, k):
        """Construct first-guess thresholds based on number of
        thresholds (k) and constraining intensity values. FirstGuesses
        will be centered around middle intensity value.
        """
        kHalf = k // 2
        midway = self.bins // 2
        firstGuesses = [midway - i for i in range(kHalf, 0, -1)] + [midway] + \
            [midway + i for i in range(1, kHalf)]
        # print("firstGuesses:",firstGuesses)
        # additional threshold in case k is odd
        firstGuesses.append(self.bins - 1)
        return firstGuesses[:k]

    def apply_thresholds_to_image(self, thresholds, im=None):
        if im is None:
            im = self.im
        k = len(thresholds)
        bookendedThresholds = [None] + thresholds + [None]
        # I think you need to use 255 / k *...
        greyValues = [0] + [int(256 / k * (i + 1)) for i in range(0, k - 1)] \
            + [255]
        greyValues = np.array(greyValues, dtype=np.uint8)
        finalImage = np.zeros(im.shape, dtype=np.uint8)
        for i in range(k + 1):
            kSmall = bookendedThresholds[i]
            # True portions of bw represents pixels between the two thresholds
            bw = np.ones(im.shape, dtype=np.bool8)
            if kSmall:
                bw = (im >= kSmall)
            kLarge = bookendedThresholds[i + 1]
            if kLarge:
                bw &= (im < kLarge)
            greyLevel = greyValues[i]
            # apply grey-color to black-and-white image
            greyImage = bw * greyLevel
            # add grey portion to image. There should be no overlap between
            # each greyImage added
            finalImage += greyImage
        return finalImage


class _ThresholdHunter(object):
    """Hunt/deviate around given thresholds in a small region to look
    for a better threshold
    """

    def __init__(self, omegas, mus, deviate=2):
        self.sigmaB = _BetweenClassVariance(omegas, mus)
        # used to be called L
        self.bins = self.sigmaB.bins
        # hunt 2 (or other amount) to either side of thresholds
        self.deviate = deviate

    def find_best_thresholds_around_estimates(self, estimatedThresholds):
        """Given guesses for best threshold, explore to either side of
        the threshold and return the best result.
        """
        bestResults = (
            0, estimatedThresholds, [0 for t in estimatedThresholds]
        )
        # print("bestResults:",bestResults)
        bestThresholds = estimatedThresholds
        bestVariance = 0
        for thresholds in self._jitter_thresholds_generator(
                estimatedThresholds, 0, self.bins):
            # print("thresholds:",thresholds)
            variance = self.sigmaB.get_total_variance(thresholds)
            if variance == bestVariance:
                if sum(thresholds) < sum(bestThresholds):
                    # keep lowest average set of thresholds
                    bestThresholds = thresholds
            elif variance > bestVariance:
                bestVariance = variance
                bestThresholds = thresholds
        return bestThresholds

    def find_best_thresholds_around_estimates_experimental(self, estimatedThresholds):
        """Experimental threshold hunting uses scipy optimize method.
        Finds ok thresholds but doesn't work quite as well
        """
        estimatedThresholds = [int(k) for k in estimatedThresholds]
        if sum(estimatedThresholds) < 10:
            return self.find_best_thresholds_around_estimates(
                estimatedThresholds
            )
        # print('estimated', estimatedThresholds)
        fxn_to_minimize = lambda x: -1 * self.sigmaB.get_total_variance(
            [int(k) for k in x]
        )
        bestThresholds = scipy.optimize.fmin(
            fxn_to_minimize, estimatedThresholds
        )
        bestThresholds = [int(k) for k in bestThresholds]
        # print('bestTresholds', bestThresholds)
        return bestThresholds

    def _jitter_thresholds_generator(self, thresholds, min_, max_):
        pastThresh = thresholds[0]
        # print("pastThresd:",pastThresh)
        if len(thresholds) == 1:
            # -2 through +2
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                if thresh < min_ or thresh >= max_:
                    # skip since we are conflicting with bounds
                    continue
                yield [thresh]
        else:
            # new threshold without our threshold included
            thresholds = thresholds[1:]
            # number of threshold left to generate in chain
            m = len(thresholds)
            for offset in range(-self.deviate, self.deviate + 1):
                thresh = pastThresh + offset
                # verify we don't use the same value as the previous threshold
                # and also verify our current threshold will not push the last
                # threshold past max
                if thresh < min_ or thresh + m >= max_:
                    continue
                recursiveGenerator = self._jitter_thresholds_generator(
                    thresholds, thresh + 1, max_
                )
                for otherThresholds in recursiveGenerator:
                    yield [thresh] + otherThresholds


class _BetweenClassVariance(object):

    def __init__(self, omegas, mus):
        self.omegas = omegas
        self.mus = mus
        # number of bins / luminosity choices
        self.bins = len(mus)
        self.muTotal = sum(mus)

    def get_total_variance(self, thresholds):
        """Function will pad the thresholds argument with minimum and
        maximum thresholds to calculate between class variance
        """
        thresholds = [0] + thresholds + [self.bins - 1]
        numClasses = len(thresholds) - 1
        # print("thesholds:",thresholds)
        sigma = 0
        for i in range(numClasses):
            k1 = thresholds[i]
            k2 = thresholds[i+1]
            sigma += self._between_thresholds_variance(k1, k2)
        # print("sigma:",sigma)
        return sigma

    def _between_thresholds_variance(self, k1, k2):
        """to be usedin calculating between-class variances only!"""
        # print("len(self.omegas)):",len(self.omegas))
        # print("k2:",k2, "kl:",k1)
        omega = self.omegas[k2] - self.omegas[k1]
        mu = self.mus[k2] - self.mus[k1]
        muT = self.muTotal
        return omega * ((mu - muT)**2)
import math 
def normalize(img):
    # 需要特别注意的是无穷大和无穷小 
    Max = -1
    Min = 10000
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > Max :
                Max = img[i][j]
            if img[i][j] < Min:
                Min = img[i][j]
    print(Max, Min)
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            if math.isnan(img[i][j]):
                img[i][j] = 255.
    img = (img - Min) / (Max - Min)
    return img * 255.

def _otsu(img, categories_pixel_nums = 1):
    '''
    ret_num: the number of image return
    img_nums: the number of calculate thresholds
    categories_pixel_nums: the number of pixel categories
    return the images with only categories_pixel_nums types of pixels
    '''
    return normalize(img.copy())
    # img = normalize(img.copy())
    # ot = OtsuFastMultithreshold()
    # ot.load_image(img)
    # kThresholds = ot.calculate_k_thresholds(categories_pixel_nums)
    # #  print(total)
    # return ot.apply_thresholds_to_image(kThresholds,img)
    # plt.imsave('C:/Users/RL/Desktop/可解释性的特征学习分类/otsu/' + str(i) + '.jpg',crushed,cmap="gray")
    # ans.append(crushed)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(img[i],cmap="gray")
    # plt.subplot(122)
    # plt.imshow(crushed,cmap="gray")
    # plt.show()

#otsu的接口，返回处理后的图片
def otsu_helper(img, upper=0.5, down = -0.5,categories=1):
    ot = OtsuFastMultithreshold()
    ot.load_image(img)
    return ot.apply_thresholds_to_image([down, upper], img)
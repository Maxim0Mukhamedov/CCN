/**
 * @file ccn.cpp
 * @brief Этот файл хранит реализацию.
 */

#include <opencv2/core.hpp>
#include "stdexcept"
#include "vector"
#include "ccn.hpp"
#include <cmath>

namespace vision {

/**
 * @brief Конвертация в RGB-linear.
 *
 * Из диапозона [0, 255] в [0, 1].
 * Функция сторонней библиотеки: https://github.com/PetterS/opencv_srgb_gamma/blob/master/srgb.h
 *
 * @param[in] srgb Число которое нужно перевести в нужный диапозон.
 * @return Возвращает конвертированное число.
 */
    inline float srgb_to_linear(float srgb) {
        auto linear = srgb / 255.0f;
        if (linear <= 0.04045f) {
            linear = linear / 12.92f;
        } else {
            linear = powf((linear + 0.055f) / 1.055f, 2.4f);
        }
        return linear;
    }

/**
 * @brief Конвертация в RGB.
 *
 * Из диапозона [0, 1] в [0, 255].
 * Функция сторонней библиотеки: https://github.com/PetterS/opencv_srgb_gamma/blob/master/srgb.h
 *
 * @param[in] linear Число которое нужно перевести в нужный диапозон.
 * @return Возвращает конвертированное число.
 */
    inline float linear_to_srgb(float linear) {
        float srgb;
        if (linear <= 0.0031308f) {
            srgb = linear * 12.92f;
        } else {
            srgb = 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
        }
        return srgb * 255.f;
    }

/**
 * @brief Конвертация в RGB-linear.
 *
 * Из диапозона [0, 255] в [0, 1].
 * Функция сторонней библиотеки: https://github.com/PetterS/opencv_srgb_gamma/blob/master/srgb.h
 *
 * @param[in] mat Изображение которое нужно перевести в нужный диапозон.
 * @return Возвращает конвертированное изображение.
 */
    static cv::Mat srgb_to_linear(const cv::Mat& mat) {
        cv::Mat linear;
        mat.convertTo(linear, CV_32FC3);
        for (auto itr = linear.begin<cv::Vec<float, 3>>(); itr != linear.end<cv::Vec<float, 3>>();
        ++itr) {
            for (int channel = 0; channel < 3; ++channel) {
                (*itr)[channel] = srgb_to_linear((*itr)[channel]);
            }
        }
        return linear;
    }

/**
 * @brief Конвертация в RGB-linear.
 *
 * Из диапозона [0, 1] в [0, 255].
 * Функция сторонней библиотеки: https://github.com/PetterS/opencv_srgb_gamma/blob/master/srgb.h
 *
 * @param[in] linear Изображение которое нужно перевести в нужный диапозон.
 * @return Возвращает конвертированное изображение.
 */
    static cv::Mat linear_to_srgb(const cv::Mat& linear) {
        cv::Mat srgb(linear.rows, linear.cols, CV_8UC3);
        for (int r = 0; r < linear.rows; ++r) {
            for (int c = 0; c < linear.cols; ++c) {
                auto& dest = srgb.at<cv::Vec<unsigned char, 3>>(r, c);
                auto& source = linear.at<cv::Vec<float, 3>>(r, c);
                for (int channel = 0; channel < 3; ++channel) {
                    dest[channel] = cv::saturate_cast<unsigned char>(linear_to_srgb(source[channel]) + 0.5f);
                }
            }
        }
        return srgb;
    }
}

namespace ccn {
/**
 * @brief Проверяет размеры и тип изображений.
 *
 * @param[in] img1 Первое изображение.
 * @param[in] img2 Второе изображение.
 * @return Возвращает True, если размер и тип изображений равны.
 */
    bool CheckImageSizeAndType(const cv::Mat &img1, const cv::Mat &img2) {

        // Проверка на размер.
        if (img1.size() != img2.size()) {
            return false;
        }

        // Проверка на тип.
        if (img1.type() != img2.type()) {
            return false;
        }
        //Вернуть True
        return true;
    }

/**
 * @brief Копирование интенсивности пикселей на основе оригинального изображения.
 *
 * @param[in] newImg Новое изображение.
 * @param[in] originImg Оригинальное изображение
 * @return Возвращет измененное изображение с учетом интенсивности пикселей оригинального изображения.
 *
 * @throws std::invalid_argument Если тип или размер изображений не равны.
 */
    cv::Mat CopyLightness(cv::Mat newImg, cv::Mat originImg) {

        // Проверка на размер и тип.
        if (!CheckImageSizeAndType(newImg, originImg)) {
            throw std::invalid_argument("type1 != type2 or size1 != size2");
        }

        // Конвертация в RGB - linear.
        newImg = vision::srgb_to_linear(newImg);
        originImg = vision::srgb_to_linear(originImg);

        // Создание изображение, в котором будет сохранен результат
        cv::Mat resultImg = newImg.clone();

        // For each pixel in the image...
        for (int x = 0; x < newImg.rows; x++) {
            for (int y = 0; y < newImg.cols; y++) {

                // Подсчет интенсивности оригинального и нового изображения
                double lNew = newImg.at<cv::Vec3f>(x, y)[0] + newImg.at<cv::Vec3f>(x, y)[1] + newImg.at<cv::Vec3f>(x, y)[2];
                double lOld = originImg.at<cv::Vec3f>(x, y)[0] + originImg.at<cv::Vec3f>(x, y)[1] + originImg.at<cv::Vec3f>(x, y)[2];

                // Изменение интенсивности нового изображения
                resultImg.at<cv::Vec3f>(x, y)[0] = newImg.at<cv::Vec3f>(x, y)[0] * (lOld / lNew);
                resultImg.at<cv::Vec3f>(x, y)[1] = newImg.at<cv::Vec3f>(x, y)[1] * (lOld / lNew);
                resultImg.at<cv::Vec3f>(x, y)[2] = newImg.at<cv::Vec3f>(x, y)[2] * (lOld / lNew);
            }
        }

        // Конвертация из RGB - linear в RGB
        resultImg = vision::linear_to_srgb(resultImg);

        //Вернуть изображение
        return resultImg;
    }

/**
 * @brief Подсчитывает среднюю разницу между изображениями.
 *
 * @param[in] img1 Первое изображение.
 * @param[in] img2 Второе изоброжение.
 * @return Средняя разнца.
 *
 * @throws std::invalid_argument Если тип или размер изображений не равны.
 */
    double CalculateAverageDiff(const cv::Mat &img1, const cv::Mat &img2) {

        // Проверка на размер и тип.
        if (!CheckImageSizeAndType(img1, img2)) {
            throw std::invalid_argument("type1 != type2 or size1 != size2");
        }

        // Подсчет общей разницы изображения.
        double totalDiff = 0.0;
        for (int i = 0; i < img1.rows; i++) {
            for (int j = 0; j < img1.cols; j++) {
                cv::Vec3b diff = img1.at<cv::Vec3b>(i, j) - img2.at<cv::Vec3b>(i, j);
                totalDiff += cv::norm(diff);
            }
        }

        // Подсчет средней разницы изображения.
        double averageDiff = totalDiff / (img1.rows * img1.cols);

        // Вернуть среднее значение.
        return averageDiff;
    }

/**
 * @brief Устранение зависимости от цвета источника освещения.
 *
 * Алгоритм GWA нормализирует значение каждого цветового канала отдельно так,
 * что бы сумма цветовых составляющих равнялась одной трети
 * количества пикселей.
 *
 * @param[in,out] img Входное изображение. Выход будет соответствовать определению GWA.
 *
 * @throws std::invalid_argument Если входное изображение пусто.
 */
    void GrayWorldAssumption(cv::Mat &img) {

        // Проверка на пустоту входного изображения.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Конвертирование в RGB-linear.
        img = vision::srgb_to_linear(img);

        // Разделение изображения на каналы.
        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        // Подсчет среднего каждого канала.
        double avgB = cv::mean(channels[0])[0];
        double avgG = cv::mean(channels[1])[0];
        double avgR = cv::mean(channels[2])[0];

        // Разделить каждый канал на среднее.
        channels[0] /= avgB * 3;
        channels[1] /= avgG * 3;
        channels[2] /= avgR * 3;

        // Восстановление изображение из каналов.
        cv::merge(channels, img);

        // Конвертирование в RGB.
        img = vision::linear_to_srgb(img);
    }

/**
 * @brief Цветовая нормализация.
 *
 * Эта функция нормализует цвета, деля каждый цветовой канал на сумму цветовых каналов.
 *
 * @param[in,out] img Входное изображение. Выход будет нормализован согласно определению.
 *
 * @throws std::invalid_argument Если входное изображение пусто.
 */
    void NormalizeColors(cv::Mat &img) {

        // Проверка на пустоту входного изображения.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Конвертирование в RGB-linear.
        img = vision::srgb_to_linear(img);

        // Разделение изображения на каналы.
        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        // Подсчет суммы каналов.
        cv::Mat sum = channels[0] + channels[1] + channels[2];

        // Деление каждого канала на сумму.
        channels[0] /= sum;
        channels[1] /= sum;
        channels[2] /= sum;

        // Восстановление изображение из каналов.
        cv::merge(channels, img);

        // Конвертирование в RGB.
        img = vision::linear_to_srgb(img);
    }



/**
* @brief Цветовая нормализация методом CCN.
*
* CCN - итеративный алгоритм, который работает в два этапа до сходимости.
* 1) Нормализация
* 2) GWA
*
* @param[in] img Входное изображение.
* @param[in] convValue Значение максимальной средней разницы между изображениями.
* @param[in] CL Использовать или нет CopyLightness алгоритм.
*
* @return Изображение нормализованное по цветам.
*
* @throws std::invalid_argument Если изображение пусто.
*/
    cv::Mat ComprColorImageNorm(const cv::Mat &img, const double &convValue, const bool &CL) {

        // Проверка на пустоту входного изображения.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Результат.
        cv::Mat resultImg = img.clone();

        // Последний шаг, нужен для подсчета разницы.
        cv::Mat lastStepImg = resultImg;

        // Итерироваться до сходимости.
        for (double stepDiff = convValue; stepDiff >= convValue; stepDiff = CalculateAverageDiff(lastStepImg, resultImg)) {

            // Нормализация каналов изображения.
            NormalizeColors(resultImg);

            // Приминение GWA.
            GrayWorldAssumption(resultImg);

            // Сохранение последнего результата.
            lastStepImg = resultImg;
        }

        // Приминение CopyLightness алгоритма.
        if (CL) {
            resultImg = CopyLightness(resultImg,img);
        }
        //Сохранить результа
        return resultImg;
    }

/**
* @brief Подсчитвает разницу по модулю между изображениями, сохраняя её в матрицу.
*
* Эта функция вычитает из первого изображения второе и сохраняет результат по модулю.
*
* @param[in] img1 Первое изображение.
* @param[in] img2 Второе изображение.
*
* @return Выход - матрица разницы между изображениями.
*
* @throws std::invalid_argument Если тип или размер изображений не равны.
*/
    cv::Mat CalculateDiff(const cv::Mat &img1, const cv::Mat &img2) {

        // Проверка на размер и тип.
        if (!CheckImageSizeAndType(img1, img2)) {
            throw std::invalid_argument("type1 != type2 or size1 != size2");
        }

        // Создание нового изображения, которое будет хранить разницу.
        cv::Mat dif;

        // Подсчет абсолютной разницы между изображениями.
        cv::absdiff(img1, img2, dif);

        // Вернуть разницу.
        return dif;
    }


/**
* @brief Цветовая нормализация методом CCN с сохранением прогрессии.
*
* CCN - итеративный алгоритм, который работает в два этапа до сходимости.
* 1) Нормализация
* 2) GWA
*
* @param[in] img Входное изображение.
* @param[in] convValue Значение максимальной средней разницы между изображениями.
*
* @return Возвращает вектор изображений.
*
* @throws std::invalid_argument Если изображение пусто.
*/
    std::vector<cv::Mat> ComprColorImageNormDiff(const cv::Mat &img, const double &lr) {
        // Проверка на пустоту входного изображения.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Результат.
        cv::Mat resultImg = img.clone();

        // Последний шаг, нужен для подсчета разницы.
        cv::Mat lastStepImg = resultImg;

        // Вектор разниц.
        std::vector<cv::Mat> diff;

        // Итерироваться до сходимости.
        for (double stepDiff = lr; stepDiff >= lr; stepDiff = CalculateAverageDiff(lastStepImg, resultImg)) {

            // Сохранение последнего результата.
            lastStepImg = resultImg;

            // Нормализация каналов изображения.
            NormalizeColors(resultImg);

            // Приминение GWA.
            GrayWorldAssumption(resultImg);

            // Подсчет и сохранение разницы
            diff.push_back(CalculateDiff(CopyLightness(lastStepImg, img), CopyLightness(resultImg, img)));
        }
        // Вернуть вектор разниц.
        return diff;
    }
}
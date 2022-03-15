#include <iostream>
#include <ostream>
#include <fstream>

#include <chrono>


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>



#include "jxl/encode.h"
#include "jxl/encode_cxx.h"

#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/resizable_parallel_runner.h"
#include "jxl/resizable_parallel_runner_cxx.h"

#include <QDir>
#include <QFileInfoList>
#include <QDebug>
#include <QtConcurrent>


bool WriteFile(const std::vector<uint8_t>& bytes, const std::string filename) {
    std::ofstream str(filename.c_str(), std::ios::binary | std::ios::out);

    if (!str)
    {
        std::cerr << "Could not open " << filename << " for writing" << std::endl;
        return false;
    }
    str.write((char*)bytes.data(), sizeof(uint8_t) * bytes.size());
    if (!str) {
        std::cerr << "Could not write bytes to" <<filename <<std::endl;
        return false;
    }
    
    return true;
}


int compress(const cv::Mat& im, std::vector<uint8_t>& compressed)
{

    auto enc = JxlEncoderMake(nullptr);
    auto runner = JxlThreadParallelRunnerMake(
        /*memory_manager=*/nullptr,
        JxlThreadParallelRunnerDefaultNumWorkerThreads());

    if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(enc.get(),
        JxlThreadParallelRunner,
        runner.get())) {
        std::cerr << "JxlEncoderSetParallelRunner failed" << std::endl;
        return -1;
    }

    JxlPixelFormat pixel_format = { 1, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0 };
    JxlBasicInfo basic_info;
    basic_info.uses_original_profile = JXL_TRUE;
    JxlEncoderInitBasicInfo(&basic_info);
    basic_info.xsize = im.cols;
    basic_info.ysize = im.rows;

    basic_info.bits_per_sample = 16;
// //    basic_info.exponent_bits_per_sample = 8;
    basic_info.uses_original_profile = JXL_FALSE; // W
    basic_info.num_color_channels = 1;
//    basic_info.uses_original_profile = JXL_TRUE; 

    if (JXL_ENC_SUCCESS != JxlEncoderSetBasicInfo(enc.get(), &basic_info)) {
        std::cerr << "JxlEncoderSetBasicInfo failed" << std::endl;
        return -1;
    }



    JxlColorEncoding color_encoding = { };

    JxlColorEncodingSetToLinearSRGB(&color_encoding, true);
    color_encoding.color_space = JXL_COLOR_SPACE_GRAY;
    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetColorEncoding(enc.get(), &color_encoding)) {
        std::cerr << "JxlEncoderSetColorEncoding failed" << std::endl;
        return -1;
    }

    //color_encoding.u
        

    JxlEncoderFrameSettings* frame_settings =
        JxlEncoderFrameSettingsCreate(enc.get(), nullptr);


    JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, 6);

    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetFrameLossless(frame_settings, true))
        std::cerr << "Warning Lossless mode not set!" << std::endl;

    //JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_MODULAR, 1); // force lossless
    //JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, 1);
    ///JxlEncoderSetFrameDistance(frame_settings,  1e-9);



    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(frame_settings, &pixel_format,
        (void*)im.ptr(), im.total() * sizeof(unsigned short int))) {
        std::cerr << "JxlEncoderAddImageFrame failed" << std::endl;
        return -1;
    }
    JxlEncoderCloseInput(enc.get());


    compressed.resize(64);
    uint8_t* next_out = compressed.data();
    size_t avail_out = compressed.size() - (next_out - compressed.data());
    JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
    while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        process_result = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
        if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
            size_t offset = next_out - compressed.data();
            compressed.resize(compressed.size() * 2);
            next_out = compressed.data() + offset;
            avail_out = compressed.size() - offset;
        }
    }
    compressed.resize(next_out - compressed.data());
    if (JXL_ENC_SUCCESS != process_result) {
        std::cerr << "JxlEncoderProcessOutput failed" << std::endl;
        return -1;
    }
    return 0;
}



int decode(std::vector<uint8_t>& comp, cv::Mat& reconstructed)
{

    std::vector<uint8_t> icc_profile;
      // Multi-threaded parallel runner.
        auto runner = JxlResizableParallelRunnerMake(nullptr);

        auto dec = JxlDecoderMake(nullptr);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
            JXL_DEC_COLOR_ENCODING |
            JXL_DEC_FULL_IMAGE)) {
            fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
            return false;
        }

        if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
            JxlResizableParallelRunner,
            runner.get())) {
            fprintf(stderr, "JxlDecoderSetParallelRunner failed\n");
            return false;
        }

        JxlBasicInfo info;
        JxlPixelFormat format = { 1, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0 };

        uint8_t* jxl = comp.data();

        JxlDecoderSetInput(dec.get(), jxl, comp.size());
        JxlDecoderCloseInput(dec.get());
        int xsize, ysize;
        std::vector<unsigned short> pixels;

        for (;;) {
            JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

            if (status == JXL_DEC_ERROR) {
                fprintf(stderr, "Decoder error\n");
                return false;
            }
            else if (status == JXL_DEC_NEED_MORE_INPUT) {
                fprintf(stderr, "Error, already provided all input\n");
                return false;
            }
            else if (status == JXL_DEC_BASIC_INFO) {
                if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
                    fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
                    return false;
                }
                xsize = info.xsize;
                ysize = info.ysize;
                std::cout << "Data " << xsize << " " << ysize << std::endl;
                JxlResizableParallelRunnerSetThreads(
                    runner.get(),
                    JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
            }
            else if (status == JXL_DEC_COLOR_ENCODING) {
           // Get the ICC color profile of the pixel data
                size_t icc_size;
                if (JXL_DEC_SUCCESS !=
                    JxlDecoderGetICCProfileSize(
                    dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
                    fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
                    return false;
                }
                icc_profile.resize(icc_size);
                if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                    dec.get(), &format,
                    JXL_COLOR_PROFILE_TARGET_DATA,
                    icc_profile.data(), icc_profile.size())) {
                    fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
                    return false;
                }
            }
            else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
                size_t buffer_size;
                if (JXL_DEC_SUCCESS !=
                    JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
                    fprintf(stderr, "JxlDecoderImageOutBufferSize failed\n");
                    return false;
                }
                if (buffer_size != xsize * ysize * 2) {
                    std::cerr << "Invalid out buffer size %" << static_cast<uint64_t>(buffer_size) << " %" << static_cast<uint64_t>(xsize * ysize * 2) << std::endl;
                    return false;
                }

                pixels.resize(xsize * ysize);
                void* pixels_buffer = (void*)pixels.data();
                size_t pixels_buffer_size = pixels.size();
                if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format,
                    pixels_buffer,
                    pixels_buffer_size*2)) {
                    fprintf(stderr, "JxlDecoderSetImageOutBuffer failed\n");
                    return false;
                }
            }
            else if (status == JXL_DEC_FULL_IMAGE) {
           // Nothing to do. Do not yet return. If the image is an animation, more
           // full frames may be decoded. This example only keeps the last one.
            }
            else if (status == JXL_DEC_SUCCESS) {
           // All decoding successfully finished.
           // It's not required to call JxlDecoderReleaseInput(dec.get()) here since
           // the decoder will be destroyed.
                cv::Mat(ysize, xsize, CV_16U, pixels.data()).copyTo(reconstructed);
                return true;
            }
            else {
                fprintf(stderr, "Unknown decoder status\n");
                return false;
            }
        }


    
    }


int compressImage(QString imf, QString tgt)
{
    auto start = std::chrono::high_resolution_clock::now();

        cv::Mat im = cv::imread(imf.toStdString(), cv::IMREAD_ANYDEPTH);
        qDebug() << "Loaded " << imf << " " << im.rows << " " << im.cols;
        auto stop = std::chrono::high_resolution_clock::now();

        float us =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
        qDebug() << QString("Tif Load time %1 s").arg(us / 1000000., 0, 'f', 2);


        std::vector<uint8_t> compressed;

        start = std::chrono::high_resolution_clock::now();
        
        if (0 == compress(im, compressed))
        {

            size_t  orig = im.total() * sizeof(unsigned short int),
                final = compressed.size();
            std::cout << "Compressing:" << orig << " to " << final << " ratio (" << 100. * (final / (double)orig) << "% )" << std::endl;
            auto stop = std::chrono::high_resolution_clock::now();
            
                float us =
                    std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                    .count();

                qDebug() << QString("Compression time %1 s").arg(us / 1000000., 0, 'f', 2);

            WriteFile(compressed, tgt.toStdString());

            cv::Mat rec;
            start = std::chrono::high_resolution_clock::now();
            decode(compressed, rec);
            stop = std::chrono::high_resolution_clock::now();

             us =
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();

             qDebug() << QString("Decompression time %1 s").arg(us / 1000000., 0, 'f', 2);

            cv::absdiff(rec, im, rec);
            cv::multiply(rec, rec, im);

            double min, max;
            cv::minMaxLoc(rec, &min, &max);
            //        cv::multiply(rec, rec, im);
            qDebug() << "Reconstruction error" << cv::sum(rec)[0] << cv::sum(rec)[0] / (double)rec.total() 
                << 10 *  log10(((double)UINT16_MAX * (double)UINT16_MAX)/ (cv::sum(im)[0]/(double)rec.total())) << max;
            return 0;
        }
        else
        {
            std::cerr << "Not Writing file due to previous errors" << std::endl;
            return -1;
        }

}

void compressImageFile(QString file)
{
    QStringList pth = file.split("/");
    
    QString tgt = QString("L:/Temp/%1/%2.jxl").arg(pth[pth.size() - 2], pth.back().chopped(4));
    //qDebug() << "Will Compress" << file << "to" << tgt;

    compressImage(file, tgt);
}

int main(int ac, const char** av)
{

    if (ac == 3)
    {
        return compressImage(QString(av[1]),av[2]); 
    }
    if (ac == 2)
    {
        std::cout << av[0] << " " << av[1] << std::endl;

        QDir dir(av[1]);

        QString flder = dir.absolutePath();
        QStringList pth = flder.split("/");
        qDebug() << "Target Folder" << QString("L:/Temp/%1").arg(pth.back());
        dir.mkpath(QString("L:/Temp/%1").arg(pth.back()));

        QFileInfoList ff = dir.entryInfoList(QStringList() << "*.tif", QDir::Files);

        QStringList tifs;
        for (auto f : ff)
        {
            tifs << f.absoluteFilePath();

        }
    //    qDebug() << tifs;

        QtConcurrent::blockingMap(tifs, compressImageFile);

        return 0;

    }

     std::cout << av[0] << " infile.tif outfile.jxl" << std::endl;
     return -1;


}


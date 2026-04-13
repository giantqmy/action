#include "postprocess_cuda.hpp"
#include <algorithm>
#include <math.h>

#define ZQ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ZQ_MIN(a, b) ((a) < (b) ? (a) : (b))

static inline float IOU(float XMin1, float YMin1, float XMax1, float YMax1, float XMin2, float YMin2, float XMax2, float YMax2)
{
    float Inter = 0;
    float Total = 0;
    float XMin = ZQ_MAX(XMin1, XMin2);
    float YMin = ZQ_MAX(YMin1, YMin2);
    float XMax = ZQ_MIN(XMax1, XMax2);
    float YMax = ZQ_MIN(YMax1, YMax2);

    float InterWidth = XMax - XMin;
    float InterHeight = YMax - YMin;
    InterWidth = (InterWidth >= 0) ? InterWidth : 0;
    InterHeight = (InterHeight >= 0) ? InterHeight : 0;

    Inter = InterWidth * InterHeight;
    float Area1 = (XMax1 - XMin1) * (YMax1 - YMin1);
    float Area2 = (XMax2 - XMin2) * (YMax2 - YMin2);
    Total = Area1 + Area2 - Inter;

    return float(Inter) / float(Total);
}

GetResultRectYolov11::GetResultRectYolov11()
{
    CoordIndex = MapSize[0][0] * MapSize[0][1] + MapSize[1][0] * MapSize[1][1] + MapSize[2][0] * MapSize[2][1];
}

GetResultRectYolov11::~GetResultRectYolov11()
{
}

int GetResultRectYolov11::GetConvDetectionResult(DetectRect *OutputRects, int *OutputCount, std::vector<float> &DetectiontRects)
{
    int ret = 0;
    std::vector<DetectRect> detectRects;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;

    DetectRect temp;
    for (int i = 0; i < *OutputCount; i++)
    {
        xmin = OutputRects[i].xmin;
        ymin = OutputRects[i].ymin;
        xmax = OutputRects[i].xmax;
        ymax = OutputRects[i].ymax;

        xmin = xmin > 0 ? xmin : 0;
        ymin = ymin > 0 ? ymin : 0;
        xmax = xmax < InputW ? xmax : InputW;
        ymax = ymax < InputH ? ymax : InputH;

        temp.xmin = xmin / InputW;
        temp.ymin = ymin / InputH;
        temp.xmax = xmax / InputW;
        temp.ymax = ymax / InputH;
        temp.classId = OutputRects[i].classId;
        temp.score = OutputRects[i].score;
        detectRects.push_back(temp);
    }

    std::sort(detectRects.begin(), detectRects.end(), [](DetectRect &Rect1, DetectRect &Rect2) -> bool
              { return (Rect1.score > Rect2.score); });

    for (int i = 0; i < static_cast<int>(detectRects.size()); ++i)
    {
        float xmin1 = detectRects[i].xmin;
        float ymin1 = detectRects[i].ymin;
        float xmax1 = detectRects[i].xmax;
        float ymax1 = detectRects[i].ymax;
        int classId = detectRects[i].classId;
        float score = detectRects[i].score;

        if (classId != -1)
        {
            DetectiontRects.push_back(float(classId));
            DetectiontRects.push_back(float(score));
            DetectiontRects.push_back(float(xmin1));
            DetectiontRects.push_back(float(ymin1));
            DetectiontRects.push_back(float(xmax1));
            DetectiontRects.push_back(float(ymax1));

            for (int j = i + 1; j < static_cast<int>(detectRects.size()); ++j)
            {
                float xmin2 = detectRects[j].xmin;
                float ymin2 = detectRects[j].ymin;
                float xmax2 = detectRects[j].xmax;
                float ymax2 = detectRects[j].ymax;
                float iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                if (iou > NmsThresh)
                {
                    detectRects[j].classId = -1;
                }
            }
        }
    }

    return ret;
}

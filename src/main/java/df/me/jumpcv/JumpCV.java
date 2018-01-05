package df.me.jumpcv;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static java.util.stream.Collectors.toList;
import static org.bytedeco.javacpp.opencv_core.minMaxLoc;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class JumpCV {

    /**
     * 通过模板匹配的方式找到小人所在的位置矩形
     *
     * @param grayImg 游戏截图的灰度图
     * @return
     */
    public Rect findCharacter(Mat grayImg) {
        float oWidth = 1080f;
        float oHeight = 2160f;
        float oiWidth = 86f;
        float oiHeight = 217f;
        //对模板的高宽根据实际图片分辨率进行处理，兼容不同的分辨率截图
        int h = Math.round((grayImg.rows() / oHeight) * oiHeight);
        int w = Math.round((grayImg.cols() / oWidth) * oiWidth);

        Mat character = Util.readImgFromClasspath("imgs/character.png", 0);
        resize(character, character, new opencv_core.Size(w, h));
        Mat iLoc = new Mat();
        matchTemplate(grayImg, character, iLoc, TM_CCORR_NORMED);
        //匹配结果iLoc是以矩阵的形式存储图像每个点（x,y）的匹配度
        DoublePointer minVal = new DoublePointer();
        DoublePointer maxVal = new DoublePointer();
        Point min = new Point();
        Point max = new Point();
        //找出矩阵中的最大值，也就是最匹配的点
        minMaxLoc(iLoc, minVal, maxVal, min, max, null);
        return new Rect(max.x(), max.y(), character.cols(), character.rows());
    }

    /**
     * 尝试通过轮廓检测的方式识别要跳的下一个目标位置
     * 但是由于阴影的干扰，还有部分目标是与小人当前站的位置有重叠，识别效果并不好
     *
     * @param img   游戏截图的灰度图
     * @param cRect 小人所在的位置矩形
     * @return
     */
    @Deprecated
    public Rect findTarget(Mat img, Rect cRect) {
        Mat grayImg = new Mat();
        img.copyTo(grayImg);
        int usefulRowStart = Math.round(grayImg.rows() * 0.2f);
        GaussianBlur(grayImg, grayImg, new opencv_core.Size(5, 5), 0);
        Canny(grayImg, grayImg, 1, 10);
        MatVector contour = new MatVector();
        Mat hie = new Mat();
        findContours(grayImg, contour, hie, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        List<Rect> found = new ArrayList<>();
        List<Rect> usefulFound = new ArrayList<>();
        for (int i = 0; i < contour.size(); i++) {
            Rect aRect = boundingRect(contour.get(i));
            found.add(aRect);
        }
        //角色图标所在区域，比实际区域有所扩大
        Rect characterArea = new Rect(cRect.x() - 10, cRect.y() - 10, cRect.width() + 10, cRect.height() + 10);
        for (int i = 0; i < found.size(); i++) {
            Rect aRect = found.get(i);
            //如果找到的边界位于角色区域内，直接丢弃
            if (isInside(aRect, characterArea)) {
                continue;
            }
            //比角色所在区域还低的边界或高于标题栏边界的也不考虑
            if (aRect.y() >= (cRect.y() + cRect.height()) || aRect.y() <= usefulRowStart) {
                continue;
            }
            boolean inside = false;
            for (int j = 0; j < found.size(); j++) {
                Rect oRect = found.get(j);
                if (i != j) {
                    inside = isInside(aRect, oRect);
                    if (inside) {
                        break;
                    }
                }
            }
            //只有最外层的边界才被添加，嵌套的边界都丢弃掉
            if (!inside) {
                usefulFound.add(aRect);
            }
        }
        List<Rect> rects = usefulFound.stream().sorted(Comparator.comparing((rect) -> rect.y())).collect(toList());
        //位置相对最高的一个作为有效目标
        if (!rects.isEmpty()) {
            return rects.get(0);
        }
        return null;
    }

    public boolean isInside(Rect inner, Rect outter) {
        return inner.x() > outter.x() && inner.y() > outter.y() && (inner.x() + inner.width()) < (outter.x() + outter.width())
                && (inner.y() + inner.height()) < (outter.y() + outter.height());
    }


    /**
     * 利用边界检测的原理，定位出目标的上边界和下边界，然后找出目标中心
     *
     * @param img   游戏截图的灰度图
     * @param cRect 小人所在的位置矩形
     * @return
     */
    public Point findTarget2(Mat img, Rect cRect) {
        Mat grayImg = new Mat();
        img.copyTo(grayImg);
        GaussianBlur(grayImg, grayImg, new opencv_core.Size(5, 5), 0);
        //用canny算法找出截图的边缘
        Canny(grayImg, grayImg, 1, 10);
        //将小人区域全部设置为黑色，避免干扰后续找边界的过程
        eraseUnUsedArea(grayImg, cRect);
        Point center = findTargetPoint(grayImg);
        return center;
    }

    public void eraseUnUsedArea(Mat img, Rect uselessArea) {
        UByteIndexer indexer = img.createIndexer();
        int rows = img.rows();
        int cols = img.cols();
        for (int i = 0; i < rows; i++) {
            if (i >= uselessArea.y() && i <= (uselessArea.y() + uselessArea.height())) {
                for (int j = 0; j < cols; j++) {
                    if (j >= uselessArea.x() && j <= (uselessArea.x() + uselessArea.width())) {
                        indexer.put(i, j, 0, 0);
                    }
                }
            }
        }
    }


    public Point findTargetPoint(Mat img) {
        //标题区域，大致是整个截图区域的20%，此部分需要排除
        int usefulRowStart = Math.round(img.rows() * 0.2f);
        UByteIndexer indexer = img.createIndexer();
        int rows = img.rows();
        int cols = img.cols();
        int topY = 0;
        int topX = 0;
        int topXstart = 0;
        int topXend = 0;
        int buttomY = 0;
        //边缘顶点所在y为从上至下扫描的第一个不为0的像素点的行数
        //x为该行中所有连续的不为0的像素点的中点
        for (int i = usefulRowStart; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int color = indexer.get(i, j, 0, 0);
                if (color != 0) {
                    topY = i;
                    topXstart = j;
                }
                if (topY != 0 && color == 0) {
                    topXend = j - 1;
                    topX = (topXstart + topXend) / 2;
                    break;
                }
            }
            if (topY != 0 && topX != 0) {
                break;
            }
        }
        //对顶点逐步往下搜索，如果找到下一个不为0的点，即是边缘的底部
        //但是由于边缘绘制的过程中可能在上边缘和下边缘留下噪点，需要跳跃一定行数进行
        //查找，以提高准确率
        //int offset = Math.round((img.rows() / 2160f) * 70);
        //TODO:对于唱片，便利店等表面花纹比较复杂的方块，找到的下边缘会被花纹干扰，虽然提高偏移量可以解决这个问题，但是又会导致小的目标块识别错误
        int minDistance = Math.round((img.rows() / 2160f) * 80);
        for (int i = topY; i < rows; i++) {
            //指定以顶点X为中心的一个范围内搜索该行不为0的点
            int noneZeroPonits = 0;
            for (int j = topX - 5; j < topX + 5; j++) {
                if (j < 0) {
                    continue;
                }
                if (j == cols - 1) {
                    break;
                }
                int color = indexer.get(i, j, 0, 0);
                if (color != 0) {
                    noneZeroPonits++;
                }
            }
            if (noneZeroPonits <= 1 && noneZeroPonits > 0 && (i - topY) > minDistance) {
                buttomY = i;
                break;
            }
        }
        return new Point(topX, topY + (buttomY - topY) / 2);
    }

}


package df.me.jumpcv;

import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.opencv_core;

import java.io.File;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Main {

    public static void main(String[] args) {
        File source = new File(Util.class.getClassLoader().getResource("imgs/samples").getFile());
        File dest = new File("trained");
        if (!source.exists() || !source.isDirectory()) {
            throw new RuntimeException("源文件夹不存在或类型错误");
        }
        if (!dest.exists()) {
            dest.mkdirs();
        }
        JumpCV cv = new JumpCV();
        String[] ext = {"jpg", "png"};
        FileUtils.listFiles(source, ext, true).stream().forEach((file) -> {
            opencv_core.Mat screen = imread(file.getAbsolutePath());
            //为了提高测试时间，将图片尺寸进行了缩减，正式使用时直接使用原图即可
            pyrDown(screen, screen);
            pyrDown(screen, screen);

            opencv_core.Mat grayImg = new opencv_core.Mat();
            cvtColor(screen, grayImg, COLOR_BGR2GRAY);
            opencv_core.Rect character = cv.findCharacter(grayImg);
            opencv_core.Point target = cv.findTarget2(grayImg, character);
            circle(screen, target, 5, new opencv_core.Scalar(0, 0, 0, 0));
            rectangle(screen, character, new opencv_core.Scalar(0, 255, 255, 0));
            imwrite(dest.getAbsolutePath() + "\\" + file.getName(), screen);
        });
        System.out.println("训练图片已经生成到：" + dest.getAbsolutePath());
    }
}

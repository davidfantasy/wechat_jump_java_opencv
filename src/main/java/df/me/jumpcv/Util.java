package df.me.jumpcv;

import org.bytedeco.javacpp.opencv_core.Mat;

import java.io.File;
import java.net.URL;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

public class Util {

    public static Mat readImgFromClasspath(String path, Integer readType) {
        URL fileUrl = Util.class.getClassLoader().getResource(path);
        String filePath = new File(fileUrl.getFile()).getAbsolutePath();
        if (readType != null) {
            return imread(filePath, readType);
        }
        return imread(filePath);
    }

}

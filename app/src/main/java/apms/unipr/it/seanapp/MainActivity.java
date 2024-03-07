package apms.unipr.it.seanapp;


import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class MainActivity extends AppCompatActivity implements Runnable{

    private static final String TAG = "SEANAppTag";
    private static final float[] NORM_MEAN_RGB = {0.5f, 0.5f, 0.5f};

    private Bitmap imageBitmap, imageMBitmap;
    private Bitmap maskBitmap, maskColBitmap, maskIBitmap;
    private Bitmap imageResultBitmap;
    List<Float> arraylist = new ArrayList<>();
    boolean vis = false;
    private String imageName = null, maskName = null;
    private String default_mask = "28022";
    private Module mModule = null;
    private ImageView imageView, imageView2;
    private TextView textView;
    private Button buttonTransform, buttonSave;
    private ProgressBar mProgressBar;
    private Tensor style_code;
    private boolean style_loaded = false;
    private String base_path = "/Download/SEAN/";

    private Map <String, Map<String, Tensor>> styleCodeMap = new HashMap<>();

    private boolean calculating = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView) findViewById(R.id.SEANimage);
        imageView2 = (ImageView) findViewById(R.id.SEANmask);
        textView = (TextView) findViewById(R.id.stateStyleCode);
        try {
            maskBitmap = BitmapFactory.decodeStream(getAssets().open("labels/" + default_mask + ".png"));
            maskColBitmap = BitmapFactory.decodeStream(getAssets().open("vis/" + default_mask + ".png"));
            maskIBitmap = BitmapFactory.decodeStream(getAssets().open("img/" + default_mask + ".jpg"));
            maskName = default_mask;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        imageView2.setImageBitmap(maskColBitmap);
        vis = true;

        //@Override
        imageView2.setOnClickListener(v -> {
            if (!vis) {
                imageView2.setImageBitmap(maskColBitmap);
                vis = true;
            } else {
                Bitmap m = Bitmap.createScaledBitmap(maskIBitmap, 512, 512, false);
                imageView2.setImageBitmap(m);
                vis = false;
            }
        });

        buttonTransform = findViewById(R.id.buttonTransform);
        buttonSave = findViewById(R.id.buttonSave);
        buttonSave.setEnabled(false);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);

        buttonTransform.setOnClickListener(v -> {
            Context context = getApplicationContext();
            int duration = Toast.LENGTH_SHORT;
            if (imageName == null) {
                CharSequence text = "Choose an image";
                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                return;
            } else {
                if (maskName == null) {
                    CharSequence text = "Choose a mask";
                    Toast toast = Toast.makeText(context, text, duration);
                    toast.show();
                    return;
                }
            }
            buttonTransform.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            buttonTransform.setText(getString(R.string.run_model));
            calculating = true;
            Thread thread = new Thread(MainActivity.this);
            thread.start();
        });

        buttonSave.setOnClickListener(v -> {
            String im = saveImage(imageResultBitmap);
            Context context = getApplicationContext();
            int duration = Toast.LENGTH_SHORT;
            CharSequence text = "Image Saved " + im;
            Toast toast = Toast.makeText(context, text, duration);
            toast.show();
        });

        //Load the module
        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "sean_enc.ptl"));
        } catch (IOException e) {
            Log.e(TAG, "Error reading assets", e);
            finish();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_selection, menu);
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu (Menu menu) {
        if (style_loaded) {
            menu.findItem(R.id.load_style_codes).setEnabled(false);
        }
        if (calculating) {
            menu.findItem(R.id.sel_photo).setEnabled(false);
            //menu.findItem(R.id.sel_mask).setEnabled(false);
        } else {
            menu.findItem(R.id.sel_photo).setEnabled(true);
            //menu.findItem(R.id.sel_mask).setEnabled(true);
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        Context context = getApplicationContext();
        if (id == R.id.sel_photo) {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_PICK);
            gallery1ResultLauncher.launch(Intent.createChooser(intent, "Seleziona immagine"));
            return true;
        } else if (id == R.id.load_style_codes) {
            try {
                loadStyleCode(context);
                style_loaded = true;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            int duration = Toast.LENGTH_SHORT;
            CharSequence text = "Loaded Style Codes";
            Toast toast = Toast.makeText(context, text, duration);
            toast.show();
            textView.setText(R.string.loaded_style_codes);
            style_loaded = true;
            Log.d(TAG, "Style Codes Loaded");
        } else if (id == R.id.sel_mask) {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_PICK);
            gallery2ResultLauncher.launch(Intent.createChooser(intent, "Seleziona maschera"));
            return true;
        } else if (id == R.id.m1 || id == R.id.m2 || id == R.id.m3 || id == R.id.m4 || id == R.id.m5 || id == R.id.m6 || id == R.id.m7) {
            item.setChecked(true);
            if (id == R.id.m1) {
                default_mask = "28000";
            } else if (id == R.id.m2) {
                default_mask = "28022";
            } else if (id == R.id.m3) {
                default_mask = "28122";
            } else if (id == R.id.m4) {
                default_mask = "28132";
            } else if (id == R.id.m5) {
                default_mask = "28270";
            } else if (id == R.id.m6) {
                default_mask = "28293";
            } else if (id == R.id.m7) {
                default_mask = "28380";
            }
            try {
                maskBitmap = BitmapFactory.decodeStream(getAssets().open("labels/" + default_mask + ".png"));
                Log.d(TAG, "new mask: " + default_mask);
                maskColBitmap = BitmapFactory.decodeStream(getAssets().open("vis/" + default_mask + ".png"));
                maskIBitmap = BitmapFactory.decodeStream(getAssets().open("img/" + default_mask + ".jpg"));
                maskName = default_mask;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            imageView2.setImageBitmap(maskColBitmap);
            vis = true;
        }
        return super.onOptionsItemSelected(item);

    }

    private void loadStyleCode(Context context) throws IOException {

        File directory = new File(Environment.getExternalStorageDirectory() + base_path + "style_codes/" + imageName + ".jpg");
        File[] sub_dir = directory.listFiles();
        Log.d("Files", "Size: "+ sub_dir.length + ", " + sub_dir[0]);
        float[] npyValues = new float[19*512];
        long[] shape = {1, 19, 512};
        String[] dir = directory.list();
        for (int i = 0; i < 19; i++) {
            int c;
            for (c = 0; c < dir.length; c++) {
                if (dir[c].equals(Integer.toString(i))) {
                    break;
                }
            }
            if (c != dir.length) {
                try {
                    File file = new File(Environment.getExternalStorageDirectory() + base_path + "style_codes/" + imageName + ".jpg/" + i + "/ACE.npy");
                    InputStream  fis = new BufferedInputStream(new FileInputStream(file));
                    Npy npy = new Npy(fis);
                    float[] npyData = npy.floatElements();
                    for (int v = 0; v < npyData.length; v++) {
                        npyValues[i*npyData.length+v]=npyData[v];
                    }
                    Map<String, Tensor> m = new HashMap<>();
                    m.put("ACE", Tensor.fromBlob(npyData, new long[]{1, npyData.length}));
                    styleCodeMap.put(Integer.toString(i), m);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            } else {
                try (InputStream stream_mean = context.getAssets().open("mean/" + i + "/ACE.npy")) {
                    Npy npy_mean = new Npy(stream_mean);
                    float[] npyData_mean = npy_mean.floatElements();
                    for (int v = 0; v < npyData_mean.length; v++) {
                        npyValues[i*npyData_mean.length+v]=npyData_mean[v];
                    }
                    Map<String, Tensor> m = new HashMap<>();
                    m.put("ACE", Tensor.fromBlob(npyData_mean, new long[]{1, npyData_mean.length}));
                    styleCodeMap.put(Integer.toString(i), m);
                }
            }

        }
        style_code = Tensor.fromBlob(npyValues, shape);
    }

    /**
     * Selection of photo
     */
    ActivityResultLauncher<Intent> gallery1ResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == Activity.RESULT_OK) {
                        Context context = getApplicationContext();

                        Intent data = result.getData();
                        if (data != null && data.getData() != null) {
                            Uri selectedImageUri = data.getData();
                            Log.d(TAG, "image " + selectedImageUri.toString());
                            try {
                                imageBitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), selectedImageUri);
                            } catch (IOException e) {
                                Log.e(TAG, "Cannot create Bitmap of image");
                                throw new RuntimeException(e);
                            }

                            String imagePath = getPath(selectedImageUri);
                            imageName = imagePath.substring(imagePath.lastIndexOf("/")+1, imagePath.length()-4);
                            try {
                                File mediaStorageDir = new File(Environment.getExternalStorageDirectory()
                                        + base_path + "labels/" + imageName + ".png");
                                InputStream  fis = new BufferedInputStream(new FileInputStream(mediaStorageDir));
                                imageMBitmap = BitmapFactory.decodeStream(fis);
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                            imageView.setImageBitmap(imageBitmap);
                            /*try {
                                loadStyleCode(context);
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }*/
                            buttonSave.setEnabled(false);
                            int duration = Toast.LENGTH_SHORT;
                            CharSequence text = "Selected image " + imageName;
                            Toast toast = Toast.makeText(context, text, duration);
                            toast.show();
                            textView.setText(R.string.none_style_codes);
                            style_loaded = false;
                            Log.d(TAG, "image " + imageName + ": " + imagePath);
                        }
                    }
                }
            });

    /**
     * Selection of mask
     */
    ActivityResultLauncher<Intent> gallery2ResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == Activity.RESULT_OK) {
                        Context context = getApplicationContext();

                        Intent data = result.getData();
                        if (data != null && data.getData() != null) {
                            Uri selectedImageUri = data.getData();
                            String maskPath = getPath(selectedImageUri);
                            Log.d(TAG, "mask " + selectedImageUri.toString());
                            maskName = maskPath.substring(maskPath.lastIndexOf("/")+1, maskPath.length()-4);
                            if (isLabel(selectedImageUri)) {
                                try {
                                    maskBitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), selectedImageUri);
                                    File mediaStorageDir = new File(Environment.getExternalStorageDirectory()
                                            + base_path + "vis/" + imageName + ".png");
                                    InputStream  fis = new BufferedInputStream(new FileInputStream(mediaStorageDir));
                                    maskColBitmap = BitmapFactory.decodeStream(fis);
                                    File mediaStorageDir2 = new File(Environment.getExternalStorageDirectory()
                                            + base_path + "images/" + imageName + ".jpg");
                                    InputStream  fis2 = new BufferedInputStream(new FileInputStream(mediaStorageDir2));
                                    maskIBitmap = BitmapFactory.decodeStream(fis2);
                                } catch (IOException e) {
                                    Log.e(TAG, "Cannot create Bitmap of mask (label) " + maskPath);
                                    throw new RuntimeException(e);
                                }
                            } else if (isVis(selectedImageUri)){
                                try {
                                    maskColBitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), selectedImageUri);
                                    File mediaStorageDir = new File(Environment.getExternalStorageDirectory()
                                            + base_path + "labels/" + imageName + ".png");
                                    InputStream  fis = new BufferedInputStream(new FileInputStream(mediaStorageDir));
                                    maskBitmap = BitmapFactory.decodeStream(fis);
                                    File mediaStorageDir2 = new File(Environment.getExternalStorageDirectory()
                                            + base_path + "images/" + imageName + ".jpg");
                                    InputStream  fis2 = new BufferedInputStream(new FileInputStream(mediaStorageDir2));
                                    maskIBitmap = BitmapFactory.decodeStream(fis2);
                                } catch (IOException e) {
                                    Log.e(TAG, "Cannot create Bitmap of mask (vis) " + maskPath);
                                    throw new RuntimeException(e);
                                }
                            } else if (isImg(selectedImageUri)) {
                                try {
                                    maskIBitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), selectedImageUri);
                                    File mediaStorageDir = new File(Environment.getExternalStorageDirectory()
                                            + base_path + "vis/" + imageName + ".png");
                                    InputStream  fis = new BufferedInputStream(new FileInputStream(mediaStorageDir));
                                    maskColBitmap = BitmapFactory.decodeStream(fis);
                                    File mediaStorageDir2 = new File(Environment.getExternalStorageDirectory()
                                            + base_path + "labels/" + imageName + ".png");
                                    InputStream  fis2 = new BufferedInputStream(new FileInputStream(mediaStorageDir2));
                                    maskBitmap = BitmapFactory.decodeStream(fis2);
                                } catch (IOException e) {
                                    Log.e(TAG, "Cannot create Bitmap of mask (image) " + maskPath);
                                    throw new RuntimeException(e);
                                }
                            } else {
                                int duration = Toast.LENGTH_SHORT;
                                CharSequence text = "No right mask selected: " + maskPath;
                                Toast toast = Toast.makeText(context, text, duration);
                                toast.show();
                                return;
                            }
                            vis = true;
                            imageView2.setImageBitmap(maskColBitmap);
                            Log.d(TAG, "mask " + maskName + ": " + maskPath);
                            imageView2.setClickable(true);
                            buttonSave.setEnabled(false);
                            int duration = Toast.LENGTH_SHORT;
                            CharSequence text = "Selected mask " + maskName;
                            Toast toast = Toast.makeText(context, text, duration);
                            toast.show();

                        }
                    }
                }
            });

    private boolean isLabel(Uri uri) {
        if (uri.toString().contains("labels")) {
            Log.d(TAG, "contains 'labels': " + uri);
            return true;
        }
        return false;
    }

    private boolean isVis(Uri uri) {
        if (uri.toString().contains("vis")) {
            Log.d(TAG, "contains 'vis': " + uri);
            return true;
        }
        return false;
    }

    private boolean isImg(Uri uri) {
        if (uri.toString().contains("images")) {
            Log.d(TAG, "contains 'images': " + uri);
            return true;
        }
        return false;
    }

    private String getPath(Uri uri) {
        if(uri == null ) {
            return null;
        }
        String[] projection = { MediaStore.Images.Media.DATA };
        Cursor cursor = getContentResolver().query(uri, projection,
                null, null, null);
        if(cursor != null ){
            int column_index = cursor.getColumnIndexOrThrow(
                    MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            return cursor.getString(column_index);
        }
        return uri.getPath();
    }

    /**
     * Create a File for saving an image or video
     * @return File to save
     **/
    private  File getOutputMediaFile(){
        File mediaStorageDir = new File(Environment.getExternalStorageDirectory()
                + "/Download");
        Log.d(TAG, "getOutputMediaFile: " + mediaStorageDir.getPath());
        if (! mediaStorageDir.exists()){
            if (! mediaStorageDir.mkdirs()){
                return null;
            }
        }
        String timeStamp = new SimpleDateFormat("ddMMyyyy_HHmm").format(new Date());
        File mediaFile;
        String mImageName="MI_"+ timeStamp +".jpg";
        mediaFile = new File(mediaStorageDir.getPath() + File.separator + mImageName);
        Log.d(TAG, "getOutputMediaFile-mediaFile: " + mediaFile);
        return mediaFile;
    }

    /**
     * Save image
     * @param image Bitmap to save
     * @return name of file saved
     */
    private String saveImage(Bitmap image) {
        File pictureFile = getOutputMediaFile();
        if (pictureFile == null) {
            Log.d(TAG,"Error creating media file, check storage permissions: ");// e.getMessage());
            return null;
        }
        try {
            FileOutputStream fos = new FileOutputStream(pictureFile);
            image.compress(Bitmap.CompressFormat.PNG, 90, fos);
            fos.close();
        } catch (FileNotFoundException e) {
            Log.d(TAG, "File not found: " + e.getMessage());
        } catch (IOException e) {
            Log.d(TAG, "Error accessing file: " + e.getMessage());
        }
        return pictureFile.getName();
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = Files.newOutputStream(file.toPath())) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    /**
     * Denormalization of Tensor
     * @param tens Tensor to denormalize
     * @param checkVal check value of Tensor
     * @return Tensor denormalized with same shape of input
     */
    public static Tensor denormalize(Tensor tens, boolean checkVal) {
        float[] tensFloat = tens.getDataAsFloatArray();
        float[] new_tens = new float[tensFloat.length];
        long[] shape = tens.shape();
        //Log.d(TAG, "denormalize " + tensFloat.length);
        List<Float> arrayvals = new ArrayList<>();
        for (int i = 0; i < tensFloat.length; i++) {
            new_tens[i] = ((tensFloat[i]+1)/2)*255;
            if (checkVal && !arrayvals.contains(new_tens[i])) {
                arrayvals.add(new_tens[i]);
            }
        }
        String shapeS = "[";
        for (int j = 0; j < shape.length; j++) {
            shapeS = shapeS.concat(shape[j] + ", ");
        }
        shapeS = shapeS.concat("]");
        Log.d(TAG, "denormalize: len " + tensFloat.length + ", shape " + shapeS);
        /*if (checkVal) {
            Log.d(TAG, "denormalize: " + shapeS + ", " + arrayvals);
        }*/
        return Tensor.fromBlob(new_tens, shape);
    }

    /**
     * Creation of One Hot Encoded Tensor
     * @param mask initial Tensor mask of shape [1,3,256,256]
     * @param n_label number of label (=19)
     * @param width width of image (=256)
     * @param height height of image (=256)
     * @return Tensor one-hot-encoded with shape [1, n_label, width, height]
     */
    public static Tensor oneHotEncoding(Tensor mask, int n_label, int width, int height) {

        float[] maskArray = mask.getDataAsFloatArray();
        long[] shape = {1, n_label, width, height};
        float[] maskOneHot = new float[width*height*n_label];
        Arrays.fill(maskOneHot, 0f);
        ArrayList<Integer> arrayVals = new ArrayList<>();
        for (int i = 0; i < width; i++) {
            for (int k = 0; k < height; k++) {
                int val = Math.round(maskArray[i*width+k]);
                if (!arrayVals.contains(val)) {
                    arrayVals.add(val);
                }
                maskOneHot[val*256*256+i*width+k] = 1.0f;
            }
        }
        Collections.sort(arrayVals);
        Log.d(TAG, "oneHotEncoding (len " + arrayVals.size() + "): " + arrayVals);
        return Tensor.fromBlob(maskOneHot, shape);

    }

    /**
     * Create a String of mask Tensor and OneHotEncoded mask Tensor
     * @param mask initial mask
     * @param oneHot one hot encoded mask
     * @return body String of note
     */
    public String createMatrixMask(Tensor mask, Tensor oneHot) {
        String result = "";
        float[] maskArray = mask.getDataAsFloatArray();
        float[] oneHotArray = oneHot.getDataAsFloatArray();
        int rows = 60;
        for (int i = 100; i < rows+100; i++) {
            for (int j = 60; j < rows+60; j++) {
                result = result.concat(Math.round(maskArray[i*256+j])+" ");
            }
            result = result.concat("\n");
        }
        result = result.concat("\n\n");
        for (int i = 100; i < rows+100; i++) {
            for (int l = 0; l < 5; l++) {
                for (int j = 60; j < rows+60; j++) {
                    result = result.concat(Math.round(oneHotArray[l*256*256+i*256+j])+" ");
                }
                result = result.concat("\t");
            }
            result = result.concat("\n");
        }
        result = result.concat("\n\n");
        for (int i = 100; i < rows+100; i++) {
            for (int l = 5; l < 10; l++) {
                for (int j = 60; j < rows+60; j++) {
                    result = result.concat(Math.round(oneHotArray[l*256*256+i*256+j])+" ");
                }
                result = result.concat("\t");
            }
            result = result.concat("\n");
        }
        result = result.concat("\n\n");
        for (int i = 100; i < rows+100; i++) {
            for (int l = 10; l < 15; l++) {
                for (int j = 60; j < rows+60; j++) {
                    result = result.concat(Math.round(oneHotArray[l*256*256+i*256+j])+" ");
                }
                result = result.concat("\t");
            }
            result = result.concat("\n");
        }
        result = result.concat("\n\n");
        for (int i = 100; i < rows+100; i++) {
            for (int l = 15; l < 19; l++) {
                for (int j = 60; j < rows+60; j++) {
                    result = result.concat(Math.round(oneHotArray[l*256*256+i*256+j])+" ");
                }
                result = result.concat("\t");
            }
            result = result.concat("\n");
        }
        return result;
    }

    /**
     * Save body in note
     * @param sBody String to save
     */
    public void generateNoteOnSD(String sBody) {
        try {
            File root = new File(Environment.getExternalStorageDirectory() + "/Download");
            if (!root.exists()) {
                root.mkdirs();
            }
            String timeStamp = new SimpleDateFormat("ddMMyyyy_HHmm").format(new Date());
            String sFileName="_"+ timeStamp +".txt";
            File gpxfile = new File(root, sFileName);
            FileWriter writer = new FileWriter(gpxfile);
            writer.append(sBody);
            writer.flush();
            writer.close();
            Log.d(TAG, "Saved Note");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Transform float Tensor in int Tensor with same shape
     * @param floatT initial float Tensor
     * @return resulting int Tensor
     */
    public static Tensor floatToIntTensor(Tensor floatT) {
        float[] floatA = floatT.getDataAsFloatArray();
        int[] intA = new int[floatA.length];
        long[] shape = floatT.shape();
        for (int i = 0; i < floatA.length; i++) {
            intA[i] = Math.round(floatA[i]);
        }
        return Tensor.fromBlob(intA, shape);
    }

    /**
     * Creation of ARGB Bitmap image from Arraylist (with Bitmap.setPixels)
     * @param floatArray initial List of Float
     * @param width image width
     * @param height image height
     * @param denorm if denormalization is needed
     * @return resulting Bitmap
     */
    private Bitmap arrayFloatToBitmapInt(List<Float> floatArray, int width, int height, boolean denorm){

        int[] pixels = new int[width*height];
        int col_R, col_G, col_B;
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888) ;
        for (int i = 0; i < width*height; i++){
            if (denorm) {
                col_R = Math.round(((floatArray.get(i)+1)/2)*255);
                col_G = Math.round(((floatArray.get(i+width*height)+1)/2)*255);
                col_B = Math.round(((floatArray.get(i+2*width*height)+1)/2)*255);
            } else {
                col_R = Math.round(floatArray.get(i));
                col_G = Math.round(floatArray.get(i+width*height));
                col_B = Math.round(floatArray.get(i+2*width*height));
            }
            pixels[i] = Color.argb(255, col_R, col_G, col_B);
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height); ;
        return bmp ;
    }

    /**
     * Creation of Grayscale Bitmap image from Arraylist
     * @param floatArray initial List of Float
     * @param width image width
     * @param height image height
     * @param denorm if denormalization is needed
     * @return resulting Bitmap
     */
    private Bitmap arrayFloatToBitmapGrayscale(List<Float> floatArray, int width, int height, boolean denorm){

        byte alpha = (byte) 255 ;
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888) ;
        ByteBuffer byteBuffer = ByteBuffer.allocate(width*height*4*3) ;
        int i = 0, val;
        for (float value : floatArray){
            if (i == width*height)
                break;
            if (denorm)
                val = Math.round((((value+1)/2)*255));
            else
                val = Math.round(value);
            byte temValue = (byte) (val);
            byteBuffer.put(4*i, temValue) ;
            byteBuffer.put(4*i+1, temValue) ;
            byteBuffer.put(4*i+2, temValue) ;
            byteBuffer.put(4*i+3, alpha) ;
            i++ ;
        }
        bmp.copyPixelsFromBuffer(byteBuffer) ;
        return bmp ;
    }

    /**
     * Creation of ARGB Bitmap image from Arraylist (with ByteBuffer)
     * @param floatArray initial List of Float
     * @param width image width
     * @param height image height
     * @param denorm if denormalization is needed
     * @return resulting Bitmap
     */
    private Bitmap arrayFloatToBitmapBuffer(List<Float> floatArray, int width, int height, boolean denorm){

        byte alpha = (byte) 255 ;

        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888) ;

        ByteBuffer byteBuffer = ByteBuffer.allocate(width*height*4*3) ;
        int i;
        byte val_R, val_G, val_B;
        for (i = 0; i < width*height; i++){
            if (denorm) {
                val_R = (byte) (Math.round((((floatArray.get(i)+1)/2)*255)));
                val_G = (byte) (Math.round(((floatArray.get(i+width*height)+1)/2)*255));
                val_B = (byte) (Math.round(((floatArray.get(i+2*width*height)+1)/2)*255));
            } else {
                val_R = (byte) (Math.round(floatArray.get(i)));
                val_G = (byte) (Math.round(floatArray.get(i+width*height)));
                val_B = (byte) (Math.round(floatArray.get(i+2*width*height)));
            }
            byteBuffer.put(4*i, val_R) ;
            byteBuffer.put(4*i+1, val_G) ;
            byteBuffer.put(4*i+2, val_B) ;
            byteBuffer.put(4*i+3, alpha) ;
        }

        Log.d(TAG, "arrayFloatToBitmap: count " + i);
        bmp.copyPixelsFromBuffer(byteBuffer) ;
        return bmp ;
    }

    /**
     * Check label of mask
     * @param image mask selected
     */
    private void findValuesMask(Bitmap image) {
        ArrayList<Integer> arrayVals = new ArrayList<>();
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                if (!arrayVals.contains(image.getPixel(x,y))) {
                    arrayVals.add(image.getPixel(x,y));
                }
            }
        }
        Log.d(TAG, "findValuesMask (len " + arrayVals.size() + ", config " + image.getConfig().toString() + "):\n");
    }


    @Override
    public void run() {

        //Scaling and creation of Tensors
        Bitmap b = Bitmap.createScaledBitmap(imageBitmap, 256, 256, true);
        Bitmap m = Bitmap.createScaledBitmap(maskBitmap, 256, 256, false);
        Bitmap m2 = Bitmap.createScaledBitmap(imageMBitmap, 256, 256, false);

        Tensor inputImageTensor = TensorImageUtils.bitmapToFloat32Tensor(b, NORM_MEAN_RGB, NORM_MEAN_RGB);
        Tensor inputMaskTensor = TensorImageUtils.bitmapToFloat32Tensor(m, NORM_MEAN_RGB, NORM_MEAN_RGB);
        Tensor inputImageMTensor = TensorImageUtils.bitmapToFloat32Tensor(m2, NORM_MEAN_RGB, NORM_MEAN_RGB);

        findValuesMask(maskBitmap);
        findValuesMask(imageMBitmap);

        //Operations on mask: denormalization and one hot encoding
        Tensor denormInputMaskTensor = denormalize(inputMaskTensor, true);
        Tensor oneHotInputMaskTensor = oneHotEncoding(denormInputMaskTensor, 19, 256, 256);
        Tensor denormInputImageMTensor = denormalize(inputImageMTensor, true);
        Tensor oneHotInputImageMTensor = oneHotEncoding(denormInputImageMTensor, 19, 256, 256);


        Log.d(TAG, "inputTensors: " + inputImageTensor + ", " + oneHotInputMaskTensor + ", " + oneHotInputImageMTensor);

        /*
        //Creation of Note with matrix one hot encoded
        String note = createMatrixMask(denormInputMaskTensor, oneHotInputMaskTensor);
        generateNoteOnSD(note);
        */

        float[] dummy_style = new float[19*512];
        long[] shape_dummy = {1, 19, 512};
        long startTime = SystemClock.elapsedRealtime();

        Tensor outputTensor = mModule.forward(IValue.from(oneHotInputMaskTensor), IValue.from(inputImageTensor), IValue.from(oneHotInputImageMTensor)).toTensor();
        //Tensor outputTensor = mModule.forward(IValue.from(oneHotInputMaskTensor), IValue.from(inputImageTensor), IValue.from(style_code)).toTensor();

        long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d(TAG,  "inference time (ms): " + inferenceTime);

        Log.d(TAG, "outputTensor: " + outputTensor);

        //Denormalization of outputTensor
        Tensor outputTensorNormalized = denormalize(outputTensor, false);

        float[] array1 = outputTensorNormalized.getDataAsFloatArray();
        if (arraylist.size() != 0) {
            arraylist.clear();
        }
        for (float v : array1) {
            arraylist.add(v);
        }

        int size = (int) Math.round(Math.sqrt(array1.length/3));
        Log.d(TAG, "arraylist: " + array1.length + ", width/height: " + size);                  //size 196608 [3*256*256]

        Bitmap res = null;
        res = arrayFloatToBitmapInt(arraylist,256,256, false);

        /*if (grayscale) {
            res = arrayFloatToBitmapGrayscale(arraylist,256,256, false);
        } else {
            //denormalization
            res = arrayFloatToBitmapInt(arraylist,256,256, false);
        }*/

        imageResultBitmap = res;
        imageResultBitmap = Bitmap.createScaledBitmap(res, 512, 512, true);

        runOnUiThread(new Runnable() {
            @Override
            public void run()
            {
                imageView.setImageBitmap(imageResultBitmap);
                buttonTransform.setEnabled(true);
                buttonTransform.setText(getString(R.string.button_transform));
                buttonSave.setEnabled(true);
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                calculating = false;
            }
        });
    }
}


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
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
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

    private Bitmap imageBitmap, maskBitmap, maskColBitmap, imageResultBitmap;
    List<Float> arraylist = new ArrayList<>();
    boolean label = false, vis = false, grayscale = false;
    private String imagePath = null, maskPath = null, maskColPath = null;
    private String imageName = null, maskName = null;
    private String default_mask = "28022";
    private Module mModule = null;
    private ImageView imageView, imageView2;
    private Button buttonTransform, buttonSave;
    private ProgressBar mProgressBar;

    private Map <String, Map<String, Tensor>> styleCodeMap = new HashMap<>();

    private boolean calculating = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView) findViewById(R.id.SEANimage);
        imageView2 = (ImageView) findViewById(R.id.SEANmask);
        try {
            imageBitmap = BitmapFactory.decodeStream(getAssets().open("img/" + default_mask + ".jpg"));
            maskBitmap = BitmapFactory.decodeStream(getAssets().open("labels/" + default_mask + ".png"));
            maskColBitmap = BitmapFactory.decodeStream(getAssets().open("vis/" + default_mask + ".png"));
            maskName = default_mask;
            imageName = default_mask;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        imageView.setImageBitmap(imageBitmap);
        imageView2.setImageBitmap(maskColBitmap);
        vis = true;
        imageView2.setOnClickListener(new View.OnClickListener() {
            //@Override
            public void onClick(View v) {
                if (label && !vis) {
                    imageView2.setImageBitmap(maskColBitmap);
                    vis = true;
                    label = false;
                } else if (vis && !label) {
                    imageView2.setImageBitmap(maskBitmap);
                    vis = false;
                    label = true;
                }
            }
        });

        buttonTransform = findViewById(R.id.buttonTransform);
        buttonSave = findViewById(R.id.buttonSave);
        buttonSave.setEnabled(false);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);

        buttonTransform.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
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
                    }/* else if ((maskName.substring(0,5)).compareTo(imageName.substring(0,5)) == 0) {            //comparing name, not extension
                        CharSequence text = "Cannot choose mask of the same image";
                        Toast toast = Toast.makeText(context, text, duration);
                        toast.show();
                        return;
                    }*/

                }
                buttonTransform.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                buttonTransform.setText(getString(R.string.run_model));
                calculating = true;
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        buttonSave.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                String im = saveImage(imageResultBitmap);
                Context context = getApplicationContext();
                int duration = Toast.LENGTH_SHORT;
                CharSequence text = "Image Saved " + im;
                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                //buttonSave.setEnabled(false);
            }
        });

        //Load the module
        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "sean.ptl"));
            //mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "sean_scripted_optimized.ptl"));
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
        if (calculating) {
            menu.findItem(R.id.sel_photo).setEnabled(false);
            //menu.findItem(R.id.sel_mask).setEnabled(false);
            //menu.findItem(R.id.grayscale_img).setEnabled(false);
            //menu.findItem(R.id.RGB_img).setEnabled(false);
        } else {
            menu.findItem(R.id.sel_photo).setEnabled(true);
            //menu.findItem(R.id.sel_mask).setEnabled(true);
            //menu.findItem(R.id.grayscale_img).setEnabled(true);
            //menu.findItem(R.id.RGB_img).setEnabled(true);
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
        }
        /*else if (id == R.id.sel_mask) {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_PICK);
            gallery2ResultLauncher.launch(Intent.createChooser(intent, "Seleziona maschera"));
            return true;
        } else if (id == R.id.grayscale_img || id == R.id.RGB_img) {
            item.setChecked(true);
            Bitmap res = null;
            if (id == R.id.grayscale_img ) {
                if (grayscale) {
                    int duration = Toast.LENGTH_SHORT;
                    CharSequence text = "Already grayscale";
                    Toast toast = Toast.makeText(context, text, duration);
                    toast.show();
                    Log.d(TAG, "image alerady grayscale");
                } else {
                    res = arrayFloatToBitmapGrayscale(arraylist,256,256);                //w 420, h 276
                    Log.d(TAG, "Grayscale: true");
                    grayscale = true;
                }
            } else if (id == R.id.RGB_img) {
                if (grayscale) {
                    res = arrayFloatToBitmapInt(arraylist,256,256);                //w 420, h 276
                    Log.d(TAG, "Grayscale: false");
                    grayscale = false;
                } else {
                    int duration = Toast.LENGTH_SHORT;
                    CharSequence text = "Already RGB";
                    Toast toast = Toast.makeText(context, text, duration);
                    toast.show();
                    Log.d(TAG, "image already RGB");
                }
            }
            if (res != null) {
                Log.d(TAG, "Changed imageBitmap");
                imageResultBitmap = Bitmap.createScaledBitmap(res, 512, 512, true);
                imageView.setImageBitmap(imageResultBitmap);
            }
        } */
        else if (id == R.id.m1 || id == R.id.m2 || id == R.id.m3 || id == R.id.m4 || id == R.id.m5 || id == R.id.m6 || id == R.id.m7 || id == R.id.m8) {
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
            } else if (id == R.id.m8) {
                default_mask = "28528";
            }
            try {
                imageBitmap = BitmapFactory.decodeStream(getAssets().open("img/" + default_mask + ".jpg"));
                maskBitmap = BitmapFactory.decodeStream(getAssets().open("labels/" + default_mask + ".png"));
                Log.d(TAG, "new mask: " + default_mask);
                maskColBitmap = BitmapFactory.decodeStream(getAssets().open("vis/" + default_mask + ".png"));
                maskName = default_mask;
                imageName = default_mask;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            imageView.setImageBitmap(imageBitmap);
            imageView2.setImageBitmap(maskColBitmap);
            vis = true;
            label = false;
        }
        return super.onOptionsItemSelected(item);

    }

    private void loadStyleCode(Context context) throws IOException {

        String[] dir = getAssets().list("style_codes/" + imageName);
        for (int i = 0; i < 19; i++) {
            int c;
            for (c = 0; c < dir.length; c++) {
                if (dir[c].equals(Integer.toString(i))) {
                    break;
                }
            }
            if (c != dir.length) {
            //if (dir != null && dir[count].equals(Integer.toString(i))) {
                //Log.d(TAG, "loadStyleCode: " + i);
                try (InputStream stream = context.getAssets().open("style_codes/" + imageName + "/" + i + "/ACE.npy")) {
                    Npy npy = new Npy(stream);
                    float[] npyData = npy.floatElements();
                    //Log.d(TAG, "\t-> " + npyData.length);
                    /*for (int j = 0; j< 20; j++) {
                        Log.d(TAG, "\t-> "+ npyData[j]);
                    }*/
                    //Dictionary<String, Tensor> d = new Hashtable<>();
                    Map<String, Tensor> m = new HashMap<>();
                    m.put("ACE", Tensor.fromBlob(npyData, new long[]{1, npyData.length}));
                    //d.put("ACE", Tensor.fromBlob(npyData, new long[]{1, npyData.length}));
                    styleCodeMap.put(Integer.toString(i), m);
                    //styleCode.put(Integer.toString(i), d);
                }
                //count++;
            } else {
                String[] dir_mean = getAssets().list("mean");
                //Log.d(TAG, "loadStyleCode: (mean) " + i);
                try (InputStream stream_mean = context.getAssets().open("mean/" + i + "/ACE.npy")) {
                    //Log.d(TAG, "loadStyleCode: (mean) " + stream_mean.read());
                    Npy npy_mean = new Npy(stream_mean);
                    float[] npyData_mean = npy_mean.floatElements();
                    //Log.d(TAG, "\t-> " + npyData_mean.length);
                    /*for (int j = 0; j< 20; j++) {
                        Log.d(TAG, "\t-> "+ npyData[j]);
                    }*/
                    //Dictionary<String, Tensor> d = new Hashtable<>();
                    Map<String, Tensor> m = new HashMap<>();
                    m.put("ACE", Tensor.fromBlob(npyData_mean, new long[]{1, npyData_mean.length}));
                    //d.put("ACE", Tensor.fromBlob(npyData_mean, new long[]{1, npyData_mean.length}));
                    styleCodeMap.put(Integer.toString(i), m);
                    //styleCode.put(Integer.toString(i), d);
                }
            }

        }
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
                            imageView.setImageBitmap(imageBitmap);
                            imagePath = getPath(selectedImageUri);
                            imageName = imagePath.substring(imagePath.lastIndexOf("/")+1);
                            try {
                                loadStyleCode(context);
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                            buttonSave.setEnabled(false);
                            int duration = Toast.LENGTH_SHORT;
                            CharSequence text = "Selected image " + imageName;
                            Toast toast = Toast.makeText(context, text, duration);
                            toast.show();
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

                            Log.d(TAG, "mask " + selectedImageUri.toString());
                            if (isLabel(selectedImageUri)) {
                                maskPath = getPath(selectedImageUri);
                                maskName = maskPath.substring(maskPath.lastIndexOf("/")+1);
                                //maskColPath = getPath(Uri.parse(new File(changeMaskPath(selectedImageUri, false)).toString()));
                                label = true;
                                try {
                                    maskBitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), selectedImageUri);
                                } catch (IOException e) {
                                    Log.e(TAG, "Cannot create Bitmap of mask " + maskPath);
                                    throw new RuntimeException(e);
                                }
                                imageView2.setImageURI(selectedImageUri);
                                Log.d(TAG, "mask " + maskName + ": " + maskPath);
                            } else if (isVis(selectedImageUri)){
                                maskColPath = getPath(selectedImageUri);
                                maskName = maskColPath.substring(maskColPath.lastIndexOf("/")+1);
                                //maskPath = getPath(Uri.parse(new File(changeMaskPath(selectedImageUri, true)).toString()));
                                vis = true;
                                try {
                                    maskBitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), selectedImageUri);
                                } catch (IOException e) {
                                    Log.e(TAG, "Cannot create Bitmap of mask colored " + maskColPath);
                                    throw new RuntimeException(e);
                                }
                                imageView2.setImageURI(selectedImageUri);
                                Log.d(TAG, "mask " + maskName + ": " + maskColPath);
                            } else {
                                int duration = Toast.LENGTH_SHORT;
                                CharSequence text = "No right mask selected";
                                Toast toast = Toast.makeText(context, text, duration);
                                toast.show();
                                return;
                            }

                            //imageView.setClickable(true);
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
            Log.d(TAG, "contains 'labels': " + uri.toString());
            return true;
        }
        return false;
    }

    private boolean isVis(Uri uri) {
        if (uri.toString().contains("vis")) {
            Log.d(TAG, "contains 'vis': " + uri.toString());
            return true;
        }
        return false;
    }

    private String changeMaskPath(Uri uri, boolean isColored) {
        if (isColored && uri.toString().contains("vis")) {    //from vis to labels
            return uri.toString().replaceAll("vis", "labels");
        } else if (!isColored && uri.toString().contains("labels")){            //from labels to vis
            return uri.toString().replaceAll("labels", "vis");
        }
        return null;
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
        // values tens [-1,+1]
        // val = ((val + 1)/2)*255.0
        float[] tensFloat = tens.getDataAsFloatArray();
        float[] new_tens = new float[tensFloat.length];
        long[] shape = tens.shape();
        Log.d(TAG, "denormalize " + tensFloat.length);
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
        if (checkVal) {
            Log.d(TAG, "denormalize: " + shapeS + ", " + arrayvals.toString());
        }
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
        Tensor inputImageTensor = TensorImageUtils.bitmapToFloat32Tensor(b, NORM_MEAN_RGB, NORM_MEAN_RGB);
        Tensor inputMaskTensor = TensorImageUtils.bitmapToFloat32Tensor(m, NORM_MEAN_RGB, NORM_MEAN_RGB);
        findValuesMask(maskBitmap);

        //Operations on mask: denormalization and one hot encoding
        Tensor denormInputMaskTensor = denormalize(inputMaskTensor, true);
        Tensor oneHotInputMaskTensor = oneHotEncoding(denormInputMaskTensor, 19, 256, 256);

        //Tensor denormInputImageTensor = denormalize(inputImageTensor, false);

        Log.d(TAG, "inputTensors: " + inputImageTensor + ", " + oneHotInputMaskTensor);

        /*
        //Creation of Note with matrix one hot encoded
        String note = createMatrixMask(denormInputMaskTensor, oneHotInputMaskTensor);
        generateNoteOnSD(note);
        */

        /*
        //Import of style_codes
        Map<String, IValue> styleCodeIVal = new HashMap<>();
        for (String k : styleCodeMap.keySet()) {
            Map<String, IValue> style_layer = new HashMap<>();
            for (String k2: styleCodeMap.get(k).keySet()) {
                style_layer.put(k2, IValue.from(styleCodeMap.get(k).get(k2)));
            }
            styleCodeIVal.put(k, IValue.dictStringKeyFrom(style_layer));
        }
        */

        long startTime = SystemClock.elapsedRealtime();
        //Tensor outputTensor = mModule.forward(IValue.from(oneHotInputMaskTensor), IValue.from(inputImageTensor), IValue.dictStringKeyFrom(styleCodeIVal)).toTensor();
        Tensor outputTensor = mModule.forward(IValue.from(oneHotInputMaskTensor), IValue.from(inputImageTensor)).toTensor();
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
            public void run() {
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


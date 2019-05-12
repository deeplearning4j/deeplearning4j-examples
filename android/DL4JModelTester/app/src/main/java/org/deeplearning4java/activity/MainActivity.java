package com.deeplearning4java.activity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;
import com.lavajaw.deeplearning4java.R;
import com.lavajaw.deeplearning4java.commons.PrefManager;

import androidx.appcompat.widget.Toolbar;

import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import static com.lavajaw.deeplearning4java.utils.Utils.getPath;

public class MainActivity extends BaseActivity {

    private static final int FILE_SELECT_CODE = 0;

    private FloatingActionButton fab;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        textView = findViewById(R.id.modelInfo);
        fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (getModel() == null) {
                    Snackbar.make(view, "Please select DL4J model", Snackbar.LENGTH_SHORT)
                            .setAction("Action", null).show();
                } else {
                    Intent intent = new Intent(MainActivity.this, CameraActivity.class);
                    startActivity(intent);
                }
            }
        });

        loadModel(PrefManager.getModelPath(this));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    public void importModel(View view) {
        showFileChooser();
    }

    private void showFileChooser() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);

        try {
            startActivityForResult(
                    Intent.createChooser(intent, "Select DL4J model"),
                    FILE_SELECT_CODE);
        } catch (android.content.ActivityNotFoundException ex) {
            Toast.makeText(this, "Please install a File Manager.",
                    Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == FILE_SELECT_CODE) {
            if (resultCode == RESULT_OK) {
                // Get the Uri of the selected file
                Uri uri = data.getData();
                // Get the path
                String path = getPath(this, uri);
                if (path != null && path.endsWith(".zip")) {
                    loadModel(path);
                } else {
                    PrefManager.setModelPath(this, null);
                    Snackbar.make(fab, "The file extension must be .zip", Snackbar.LENGTH_SHORT)
                            .setAction("Action", null).show();
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void loadModel(String path) {
        if (path != null) {
            Snackbar.make(fab, "Loading model...", Snackbar.LENGTH_INDEFINITE)
                    .setAction("Action", null).show();
            loadDL4JModel(path, new LoadModelListener() {
                @Override
                public void loadFinished(MultiLayerNetwork multiLayerNetwork) {
                    textView.setText(getModelInfo(multiLayerNetwork));
                    PrefManager.setModelPath(MainActivity.this, path);
                    Snackbar.make(fab, "Model loaded", Snackbar.LENGTH_SHORT)
                            .setAction("Action", null).show();
                }

                @Override
                public void loadFinished(ComputationGraph computationGraph) {
                    textView.setText(getModelInfo(computationGraph));
                    PrefManager.setModelPath(MainActivity.this, path);
                    Snackbar.make(fab, "Model loaded", Snackbar.LENGTH_SHORT)
                            .setAction("Action", null).show();
                }

                @Override
                public void loadFailed(Exception e) {
                    PrefManager.setModelPath(MainActivity.this, null);
                    Snackbar.make(fab, "The file isn't DL4J model", Snackbar.LENGTH_SHORT)
                            .setAction("Action", null).show();
                    textView.setText(getString(R.string.please_choose));
                }
            });
        } else {
            textView.setText(getString(R.string.please_choose));
        }
    }

    private String getModelInfo(ComputationGraph neuralNetwork) {
        if (neuralNetwork != null) {
            return String.format("%s %s\n %s %s\n %s %s\n %s %s\n",
                    "Type: ", "ComputationGraph",
                    "Last Etl time: ", neuralNetwork.getLastEtlTime(),
                    "Number of layer: ", neuralNetwork.getLayers().length,
                    "Number of epochs: ", neuralNetwork.getEpochCount());
        }
        return getString(R.string.please_choose);
    }

    private String getModelInfo(MultiLayerNetwork neuralNetwork) {
        if (neuralNetwork != null) {
            return String.format("%s %s\n %s %s\n %s %s\n %s %s\n",
                    "Type: ", "MultiLayerNetwork",
                    "Last Etl time: ", neuralNetwork.getLastEtlTime(),
                    "Number of layer: ", neuralNetwork.getLayers().length,
                    "Number of epochs: ", neuralNetwork.getEpochCount());
        }
        return getString(R.string.please_choose);

    }
}

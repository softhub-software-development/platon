<html>
  <head>
    <meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'/>
    <meta http-equiv='Pragma' content='no-cache'/>
    <meta http-equiv='Expires' content='0'/>
    <meta name='viewport' content='width=device-width,initial-scale=1,maximum-scale=1,user-scalable=0'>
    <script>

      function showProgressMessage(msg) {
        document.getElementById("progressMessage").innerHTML = msg;
      }

      function onUpload(msg) {
        showProgressMessage("Press the upload button.");
      }

      function onSubmit(msg) {
        showProgressMessage("Please wait for the upload to complete...");
      }

    </script>
  </head>
  <body>
    <p>Example web application which uses machine learning to obfuscate license plates.</p>
    <form id='demoform' action='demo.php' method='post' enctype='multipart/form-data'>
      <p>
        <input type='file' name='fileToUpload' id='fileToUpload' onchange='onUpload()'/>
        <input type='submit' value='Upload Image' name='submit' onclick='onSubmit()'/>
        <div id="progressMessage">Choose image and press upload button.</div>
      </p>
<?php

$WWW_DIR = "/var/www/ai/";
$TARGET_DIR = "uploads/";
$MAX_IMG_SIZE = 20000000;
$MB_FACTOR = 1024 * 1024;
$CHECK_FILE_EXISTS = false;

function showImages() {
  $t = time();
  echo "      <table border='1'><tr><td>\n";
  echo "        <img width='480px' src='tmp/demo-in.png?prevent_caching=" . $t . "'/>\n";
  echo "      </td><td>\n";
  echo "        <img width='480px' src='tmp/demo-out.png?prevent_caching=" . $t . "'/>\n";
  echo "      </td></tr><tr><td align='center'>Original</td><td align='center'>Obfuscated</td></tr>\n";
  echo "      </table>\n";
}

function processUpload($args) {
# echo "The file ". basename($_FILES["fileToUpload"]["name"]). " has been uploaded. Process " . $args . ".";
  $cmd = "cd ~chris/ai/app/products/platon/; ./main.sh " . $args;
  exec($cmd, $result);
  showImages();
}

function validFileType($imageType) {
  $ext = strtolower($imageType);
  return $ext == "jpg" || $ext == "jpeg" || $ext == "png";
}

if (isset($_POST["submit"])) {
  $target_name = urlencode($_FILES["fileToUpload"]["name"]);
  $target_file = $TARGET_DIR . basename($target_name);
  $imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));
  // Check if image file is a actual image or fake image
  $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
  if ($check === false) {
    echo "File is not an image.";
  } else {
    if ($CHECK_FILE_EXISTS && file_exists($target_file)) {
      echo "Sorry, file already exists.";
    } else {
      $file_size = $_FILES["fileToUpload"]["size"];
      if ($file_size > $MAX_IMG_SIZE) {
        echo round($file_size / 1024) . " kb, your file is too large. Maximum is " . round($MAX_IMG_SIZE / $MB_FACTOR) . " mb.";
      } else {
        if (validFileType($imageFileType)) {
          if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
            processUpload($WWW_DIR . $target_file . " " . "demo-");
          } else {
            echo "Sorry, there was an error uploading " . $target_file;;
          }
        } else {
          echo "Sorry, only JPG, JPEG and PNG files are allowed.";
        }
      }
    }
  }
} else {
  showImages();
}

?>
    </form>
  </body>
</html>

<?php

#phpinfo();

$WWW_DIR = "/var/www/ai/";
$TARGET_DIR = "uploads/";
$MAX_IMG_SIZE = 20000000;
$MB_FACTOR = 1024 * 1024;
$CHECK_FILE_EXISTS = false;

function visualize($filename)
{
  $t = time();
  $src = null;
  $dst = null;
  $lic = array();
  $file = fopen($filename, "r");
  if ($file) {
    while (($line = fgets($file)) !== false) {
      list($cmd, $col1, $col2, $col3) = split(':', $line);
      if ($cmd == "src") {
        $src = $col1;
      } else if ($cmd == "dst") {
        $dst = $col1;
      } else if ($cmd == "lic") {
        $desc = array($col1, $col2, $col3);
        array_push($lic, $desc);
      }
    }
    fclose($file);
    echo "<table>\n";
    echo "<tr><td>\n";
    echo "  <img width='520px' src='" . $src . "?prevent=caching_" . $t . "'/>\n";
    echo "</td><td>\n";
    echo "  <img width='520px' src='" . $dst . "?prevent=caching_" . $t . "'/>\n";
    foreach ($lic as $license) {
      echo "<tr><td>\n";
      echo "  <img src='" . $license[0] . "?prevent=caching_" . $t . "'/>\n";
      echo "</td><td>\n";
      echo "  <img width='300px' src='" . $license[0] . "?prevent=caching_" . $t . "'/>\n";
      echo "</td></tr><tr><td>\n";
      echo "</td><td>\n";
      echo "<p>" . $license[2] . " " . $license[1] . "\n";
      echo "</td></tr>\n";
    }
    echo "</table>\n";
  } else {
    echo "Failed to open " . $filename . "\n";
  }
}

if (isset($_POST["submit"])) {
  echo "<html>";
  echo "<head>";
  echo "  <meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'/>";
  echo "  <meta http-equiv='Pragma' content='no-cache'/>";
  echo "  <meta http-equiv='Expires' content='0'/>";
  echo "</head>";
  echo "<body>";
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
        if ($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg") {
          echo "Sorry, only JPG, JPEG and PNG files are allowed.";
        } else {
          if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
            echo "The file ". basename($_FILES["fileToUpload"]["name"]). " has been uploaded.\n";
            $cmd = "cd ~chris/ai/app/products/platon/; ./main.sh " . $WWW_DIR . $target_file . " lic- debug";
            $t = "0";//time();
            exec($cmd, $result);
#           echo "done: " . print_r($result);
            visualize('tmp/lic-' . 'desc.txt');
          } else {
            echo "Sorry, there was an error uploading " . $target_file;;
          }
        }
      }
    }
  }
  echo "</body>";
  echo "</html>";
}

?>


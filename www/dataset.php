<html>
<head>
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
  <meta http-equiv="Pragma" content="no-cache"/>
  <meta http-equiv="Expires" content="0"/>
  <meta name='viewport' content='width=device-width,initial-scale=1,maximum-scale=1,user-scalable=0'>
</head>
<body>
  <table>
<?php

function imageList($sub_dir, $width) {
  $ds_dir = "dataset/" . $sub_dir . "/";
  $opt_width = $width == 1 ? "" : "width='" . $width . "px' ";
  if ($handle = opendir($ds_dir)) {
    while (false !== ($entry = readdir($handle))) {
      if ($entry != "." && $entry != "..") {
         echo "    <tr><td>\n";
         echo "      <p>$entry</p>\n";
         echo "    </td></tr><tr><td>\n";
         echo "      <img " . $opt_width . "src='" . $ds_dir . $entry . "'/>\n";
         echo "    <tr><td>\n";
         echo "      <p>&nbsp;</p>\n";
         echo "    </td></tr>\n";
      }
    }
    closedir($handle);
  }
}

$param = $_GET["dir"];

if ($param === "pos12")
  imageList("val/12/positive", 200);
else if ($param === "part12")
  imageList("val/12/part", 200);
else if ($param === "neg12")
  imageList("val/12/negative", 200);
else if ($param === "pos24")
  imageList("val/24/positive", 200);
else if ($param === "part24")
  imageList("val/24/part", 200);
else if ($param === "neg24")
  imageList("val/24/negative", 200);
else
  imageList("raw/images", 1);

?>
  </table>
</body>
</html>

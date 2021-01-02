<html>
  <head>
    <title>test</title>
    <meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'/>
    <meta http-equiv='Pragma' content='no-cache'/>
    <meta http-equiv='Expires' content='0'/>
    <meta name='viewport' content='width=device-width,initial-scale=1,maximum-scale=1,user-scalable=0'>
  </head>
  <body>
    <p>Test Results</p>
    <table>
<?php

$WWW_DIR = "/var/www/ai/";
$REL_TEST_DIR = "tmp/test/";
$ABS_TEST_DIR = $WWW_DIR . $REL_TEST_DIR;

if ($dir = opendir($ABS_TEST_DIR)) {
  while (($entry = readdir($dir))) {
    if ($entry != "." && $entry != ".." && !fnmatch("*-ori.*", $entry) && !fnmatch("*-old.*", $entry)) {
      $info = pathinfo($entry);
      $old = $REL_TEST_DIR . $info['filename'] . "-old." . $info['extension'];
      $ori = $REL_TEST_DIR . $info['filename'] . "-ori." . $info['extension'];
      $act = $REL_TEST_DIR . $entry;
      echo "      </td></tr><tr><td colspan='3'>" . $entry . "</td></tr>\n";
      echo "      <tr><td>\n";
      echo "        <img width='280px' src='" . $ori . "'/>\n";
      echo "      </td><td>\n";
      echo "        <img width='280px' src='" . $old . "'/>\n";
      echo "      </td><td>\n";
      echo "        <img width='280px' src='" . $act . "'/>\n";
      echo "      </td></tr><tr><td align='center'>orignal</td><td align='center'>old</td><td align='center'>new</td></tr>\n";
      echo "      </td></tr><tr colspan='3'><td>&nbsp;</td></tr>\n";
    }
  }
  closedir($dir);
}

?>
    </table>
  </body>
</html>

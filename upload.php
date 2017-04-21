<?php
        $total = count($_FILES['imgs']['name']);
        $response = array();  
        $curtime = $_POST['curtime'];
	// Loop through each file
	for($i=0; $i<$total; $i++) {
	  //Get the temp file path
	  $tmpFilePath = $_FILES['imgs']['tmp_name'][$i];

	  //Make sure we have a filepath
	  if ($tmpFilePath != ""){
            //get file type
            $imgtype =  substr($_FILES['imgs']['name'][$i], strrpos($_FILES['imgs']['name'][$i], ".")+1);  
	    //Setup our new file path
	    $newFilePath = "./file/" . $curtime . "-". (string)$i . "." . $imgtype;

	    //Upload the file into the temp dir
	    if(move_uploaded_file($tmpFilePath, $newFilePath)) {

	      //Handle other code here
		//$json_arr = array("username"=>$username,"age"=>$age,"job"=>$job);
		//$json_obj = json_encode($json_arr);
		$response['isSuccess'] = true;  
                $response['filename'] = $curtime;  

	    }
            else{
		$response['isSuccess'] = false;  		
		}
	  }
	}
        echo json_encode($response);  
?>

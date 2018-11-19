FEGB torso orientation control

From paper: 
"Online Learning of Body Orientation Control on a Humanoid Robot using Finite Element Goal Babbling", by Loviken P. et al (2018)

=============================================================

1. Make sure you have dependencies found in "requirements.txt"

2. Configure Nao's wireless connection and disable Nao's reflexes
	2.1.  Turn Nao on with an ethernet cable inserted.
	2.2.  Once active, press the chest button; the robot will state its IP address.
	2.3.  Access the IP address via an internet browser.
	2.4.  Enter Username and Password ("nao" by default).
        2.5.  Under Wifi settings, connect Nao to your Wifi
	2.6.  Remove the ethernet cable and press the chest button again to get the new IP address on the Wifi network.
        2.7.  Enter the new IP address via an internet browser to check that the wireless connection to Nao is working.  
	2.8.  Disable "Alive by default"
	2.9.  Disable "Fall manager", see: 
		http://doc.aldebaran.com/2-1/naoqi/motion/reflexes-fall-manager.html
	2.10. Reboot the robot

3. Open "main.py" in an editor
	Row 19 	If you have a saved previous file, write the name here without the file extension.
		This is useful as it allows longer runs where the robot can rest in-between.
	Row 20	Enter the name you want to save the results under. Set to "None" to not save. 
	Row 24	Enter the wifi IP address of the robot
	
4. Run "main.py".
	- The robot will be relaxed for 10 seconds before starting, to allow manual positioning.
	- To make the robot attempt to reach a specific state, left-click on it in the graphical interface.
	- To make the robot resume intrinsically motivated exploration, click outside the square.
	- Be careful so the robot does not fall over to the belly as this may damage the shoulders.
	- If you want to abort learning and make the robot relax, just kill the prompt, restart the program,
	  and kill it during the initial 10s during which the Nao relaxes.

5. Results can be visualized with "Display results.ipynb", with "jupyter" or similar.

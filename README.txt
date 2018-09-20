FEGB torso orientation control

From paper: 
"Online Learning of Body Orientation Control on a Humanoid Robot using Finite Element Goal Babbling", by Loviken P. et al (2018)

=============================================================

1. Make sure you have dependencies found in "requirements.txt"

2. Configurate the Nao for use
	2.1.  Turn it on with an ethernet cable inserted
	2.2.  Once active, press the chest button, the robot will state its IP address
	2.3.  Enter the IP address into a browser.
	2.4.  Username and Password is "nao" by default.
	2.5.  Remove ethernet cable and press the chest button again for the new IP address.
	2.6.  Go to the webpage of the new IP address and remove the ethernet cable.
	2.7.  Connect the robot to the same wifi network as the computer.
	2.8.  Disable "Alive by default"
	2.9.  Disable "Fall manager", see: 
		http://doc.aldebaran.com/2-1/naoqi/motion/reflexes-fall-manager.html
	2.10. Reboot the robot

3. Open "main.py" in an editor
	Row 19 	If you have a saved previous file, write the name here without the file extension.
		This is useful as it allows longer runs where the robot can rest in between.
	Row 20	Enter the name you want to save the results under. Set to "None" to not save. 
	Row 24	Enter the IP address of the robot
	
4. Run "main.py".
	- The robot will be relaxed for 10 seconds before starting, to allow manual positioning.
	- To make the robot attempt to reach a specific state, click on it.
	- To make the robot resume intrinsically motivated exploration, click outside the square.
	- Be careful so the robot does not fall over to the belly as this may damage the shoulders.
	- If you want abort learning and make the robot relax, just kill the prompt and restart,
	  and kill the prompt again once the robot relax.

5. To visualize the results, run "Display results.ipynb" with "jupyter" or similar.

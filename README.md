# AD-Honeypot
Project for using GNNs for Active Directory modeling and enrichment with a honeytokens. Based on Tensorflow 2.

## Author
Ondřej Lukáš
lukasond@fel.cvut.cz, @ondrej_lukas


## Overview
Active Directory (AD) is one of the cornerstones of internal network administration in many organizations. It holds information about users, resources, access rights and other relations within the organization's network that helps administer it.
Because of its importance, attackers have been targeting AD in order to obtain additional information for attack planning, to access sensitive data, or to get persistence and ultimately complete control of the domain. After the initial breach, the attackers commonly perform AD reconnaissance. By design, any user with basic access rights can query the AD database, which means that a password leak of even the most unprivileged user is sufficient to gather information about almost any entity within.
A common technique while attacking the AD is called lateral movement. Attackers try to explore the network of the organization without being detected. During this time, they are performing reconnaissance in the AD in order to find high-value targets and ways of getting persistence in the domain. In these attacking scenarios the use of honeypots may greatly improve the detection capabilities of the organization by providing an early warning system. Honeypots are a well-known form of passive security measures. In the most basic form, they are decoys disguised as real devices or information about a user, in this last form they are known as honeytokens. 
Despite being useful and promising a good detection, the basic constraint of a honeypot is that it should be found before the intruders attack a real target. Therefore, it is crucial to have the honeyuser placed correctly into the AD structure. However, with the complexity and diversity of AD structures, this task is very hard.

 We propose three variants of the model architecture and evaluate the performance of each them. Results show that the proposed models achieve F1 score over 0.6 in structure reconstruction tasks. Moreover, the validity ratio of the predicted placement is over 60\% for the graphs of sizes similar to the real-world AD environments.

This project is part of the Master's thesis of Ondřej Lukáš.

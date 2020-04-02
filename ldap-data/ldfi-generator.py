import random
from math import log,ceil
import sys
from ldif3 import LDIFWriter
import hashlib
import string

#python ldfi-generator.py .\Data\organizational-units.txt .\Data\given-names.txt .\Data\family-names.txt .\Data\localities.txt .\Data\mail-hosts.txt

def get_list(src_file):
	ret = []
	with open(src_file, "r") as f:
		for line in f:
			ret.append(line.rstrip("\n"))
	return ret
def generate_password(l):
	password_characters = string.ascii_letters + string.digits + string.punctuation
	return ''.join([random.choice(password_characters) for i in range(l)])

class LDIPOrganization():
	def __init__(self, org_name, filename="test_generation.ldif"):
		self.supervisors = set()
		self.org_units = set()
		self.dn_set = set()
		self.usernames = set()
		self.phone_numbers = set()
		
		self.emails = set()
		self.max_login_len = random.randint(5,12)
		self.login_version = random.randint(0,3)

		self.target_file = filename
		self.base_dn = "o="+ org_name

		#generate organization level data:
		with open(self.target_file,"wb") as f:
			writer = LDIFWriter(f)
			writer.unparse(self.base_dn, {"o":[org_name], "objectClass":["top", "organization"]})

	def generte_OUs(self, src_data, num_records, num_ou=None):
		with open(self.target_file,"ab") as f:
			writer = LDIFWriter(f)
			dn = None
			if not num_ou:
				num_ou = ceil(log(log(num_records, 1.2)))
			for _ in range(num_ou):
				while dn == None or dn in self.org_units:
					ou = random.choice(src_data)
					dn = f"ou={ou},{self.base_dn}"
				self.org_units.add(dn)
				writer.unparse(dn, {"ou":[ou], "objectClass":["top", "organizationalUnit"]})
			#generate ou 'people'
			writer.unparse(f"ou=people,{self.base_dn}",{"ou":["people"], "objectClass":["top", "organizationalUnit"]})

	def generate_phone_number(self, l, prefixes=None, country_code=None):
		ret = ""
		if country_code:
			ret += country_code
		if prefixes:
			x = random.choice(prefixes)
			x += "".join(random.choices(string.digits,k=l-len(x)))
			ret+= x
		else:
			x = "".join(random.choices(string.digits,k=l))
			ret += x
		if ret in self.phone_numbers:
			ret = self.generate_phone_number(l,prefixes,country_code)
		return ret

	def genereate_person(self, src_family_names, src_given_names, src_locations,src_mails):
		dn = None
		while dn == None or dn in self.dn_set:
			sn = random.choice(src_family_names)
			gn = random.choice(src_given_names)
			ou = random.choice(list(self.org_units)).split(",")[0].lstrip("ou=")
			cn = gn + " " + sn
			location = random.choice(src_locations)
			username = self.get_username(gn,sn)
			email = [f"{username}@{self.base_dn.lstrip('o=')}"]
			if random.random() > 0.5:
				email.append(f"{random.choice([sn,sn[0],sn.lower(), gn, gn[0], gn.lower(),gn[:random.randint(1,len(gn))],sn[:random.randint(1,len(sn))]])}{random.choice(['.','_',''])}{random.choice([sn,sn[0],sn.lower(), gn, gn[0], gn.lower(),gn[:random.randint(1,len(gn))],sn[:random.randint(1,len(sn))]])}@{random.choice(src_mails)}")
			object_classes = ["top", "person", "organizationalPerson", "inetOrgPerson"]
			password = hashlib.sha256(generate_password(random.randint(8,16)).encode()).hexdigest()
			phone_number = self.generate_phone_number(9,prefixes=["6","7"])
			dn = f"uid={username},ou={ou},{self.base_dn}"
		with open(self.target_file,"ab") as f:
			writer = LDIFWriter(f)

			data = {"cn":[cn], "objectClass":["top", "person", "organizationalPerson", "inetOrgPerson"], "sn":[sn], "givenName":[gn],"uid":[username], "email":email, "ou":[ou,'people'], "phone":[phone_number], "password":[password]}
			#print(data)
			writer.unparse(dn, data)
			self.dn_set.add(dn)
			self.usernames.add(username)
			self.phone_numbers.add(phone_number)

	def get_username(self, gn, sn):
		ret = sn.replace(" ", "")[:min(len(sn), self.max_login_len-1)]
		ret += gn.replace(" ", "")
		ret = ret[:self.max_login_len].lower()
		if ret in self.usernames:
			i = 1
			ret = f"{ret}{i}"
			while ret in self.usernames:
				i +=1 
				ret = f"{ret}{i}"
		return ret.lower()

	def generate_DB(self, num_ou, num_entries):
		print("generating:", self.base_dn)	
		#generate organisational units
		self.generte_OUs(src_org_units, num_ou, num_entries)
		#generate people
		for _ in range(num_entries):
			self.genereate_person(src_family_names, src_given_names, src_locations, src_mails)

if __name__ == '__main__':
	src_org_units = get_list(sys.argv[1])
	src_given_names = get_list(sys.argv[2])
	src_family_names = get_list(sys.argv[3])
	src_locations = get_list(sys.argv[4])
	src_mails = get_list(sys.argv[5])
	orgs = ["aic.fel.cvut.cz" ,"seznam.cz", "test.com", "google.com", "prg.ai", "cvut.cz"]
	sizes = [random.randint(10,5000) for k in range(0,len(orgs))]
	print(sizes)
	#generator = LDIPOrganization("aic.fel.cvut.cz")	
	for i in range(len(orgs)):
		generator = LDIPOrganization(orgs[i], f"data_{orgs[i]}")
		generator.generte_OUs(src_org_units,sizes[i])
		for _ in range(sizes[i]):
			generator.genereate_person(src_family_names, src_given_names, src_locations, src_mails)
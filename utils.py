#!/usr/bin/env python3
# Author: Ondrej Lukas - lukasond@fel.cvut.cz
import json
from argparse import ArgumentParser
import numpy as np
import pickle
import networkx as nx
import sys
import csv
import random

def transform_graph(A, ordering):
    new_A = np.zeros_like(A)
    mapping = {ordering[i]: i for i in range(len(ordering))}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            new_A[mapping[i], mapping[j]] = A[i, j]
    return new_A


def process_sharphound_data(args, classes={"domain": 0, "ou": 1, "container": 2, "group": 3, "user": 4, "computer": 4}):
    objects = {}

    DOMAIN_DN = "UNKNOWN"
    # read domains
    with open(args.d_file, "r", encoding="utf-8-sig") as infile:
        data = json.load(infile)
        for item in data["domains"]:
            oid = item["ObjectIdentifier"]
            dn = item["Properties"]["distinguishedname"]
            parent = None
            rdn = dn.split(',')[0]
            children = []
            children += item["Users"]
            children += item["Computers"]
            children += item["ChildOus"]
            DOMAIN_DN = dn
            aces = []
            #store acces rights
            for ace in item["Aces"]:
                aces.append(ace["PrincipalSID"])
            objects[oid] = {"dn":dn, "rdn":rdn, "aces":aces, "objectType":"domain","children":children,"parent":parent, "node_id":len(objects.keys())}

    # read users:
    with open(args.u_file, "r", encoding="utf-8-sig") as infile:
        data = json.load(infile)
        i = 0
        for item in data["users"]:
            oid = item["ObjectIdentifier"]
            dn = item["Properties"]["distinguishedname"]
            print(dn)
            parent = ",".join(dn.split(",")[1:])
            rdn = dn.split(',')[0]
            aces = []
            #store acces rights
            for ace in item["Aces"]:
                aces.append(ace["PrincipalSID"])
            objects[oid] = {"dn": dn, "rdn": rdn, "aces": aces, "objectType": "user", "parent": parent, "node_id": len(objects.keys()), "pGroupSID": item["PrimaryGroupSid"]}
            i += 1
        print(f"{i} users")

    # read computers:
    with open(args.c_file, "r", encoding="utf-8-sig") as infile:
        i = 0
        data = json.load(infile)
        for item in data["computers"]:
            oid = item["ObjectIdentifier"]
            dn = item["Properties"]["distinguishedname"]
            parent = ",".join(dn.split(",")[1:])
            rdn = dn.split(',')[0]
            aces = []
            #store acces rights
            for ace in item["Aces"]:
                aces.append(ace["PrincipalSID"])
            objects[oid] = {"dn":dn, "rdn":rdn, "aces":aces, "objectType":"computer", "spNames":item["Properties"]["serviceprincipalnames"],"parent":parent,"node_id":len(objects.keys()), "pGroupSID":item["PrimaryGroupSid"]}
            i += 1
        print(f"{i} Computers")
    # read ous:
    with open(args.o_file, "r",encoding="utf-8-sig") as infile:
        i = 0
        data = json.load(infile)
        for item in data["ous"]:
            oid = item["ObjectIdentifier"]
            dn = item["Properties"]["distinguishedname"]
            rdn = dn.split(',')[0]
            parent = ",".join(dn.split(",")[1:])

            children = []
            children += item["Users"]
            children += item["Computers"]
            children += item["ChildOus"]

            aces = []
            #store acces rights
            for ace in item["Aces"]:
                aces.append(ace["PrincipalSID"])
            objects[oid] = {"dn":dn, "rdn":rdn, "aces":aces, "objectType":"ou", "children":children, "parent":parent,"node_id":len(objects.keys())}
            i += 1
        print(f"{i} OUs")

    # read groups:
    with open(args.g_file, "r", encoding="utf-8-sig") as infile:
        i = 0
        data = json.load(infile)
        for item in data["groups"]:
            oid = item["ObjectIdentifier"]
            if oid in objects:
               continue
            if "distinguishedname" in item["Properties"].keys():
                dn = item["Properties"]["distinguishedname"]
                rdn = dn.split(',')[0]
                parent = ",".join(dn.split(",")[1:])
            else:
                dn = None
                rdn = None
                parent = DOMAIN_DN

            children = []
            for x in item["Members"]:
                children.append(x["MemberId"])

            aces = []
            #store acces rights
            for ace in item["Aces"]:
                aces.append(ace["PrincipalSID"])
            objects[oid] = {"dn": dn, "rdn": rdn, "aces": aces, "objectType": "group", "children": children, "parent":parent,"node_id": len(objects.keys())}
            i+= 1
        print(f"{i} groups")

    #prepare matrices
    X = -1*np.ones(len(objects.keys()))
    A = np.zeros([len(objects.keys()),len(objects.keys())])

    node_id_to_oid = {}
    #fix edges
    for k,o in objects.items():
        #add node type:
        X[o["node_id"]] = classes[o["objectType"]]
        node_id_to_oid[o["node_id"]] = k
        if "children" in o.keys():
            for c_oid in o["children"]:
                try:
                    A[o["node_id"], objects[c_oid]["node_id"]] = 1
                except KeyError as e:
                    pass
                    print(f"skipping: {e}")
        if o["objectType"] in ["user", "computer"]:
            if "pGroupSID" in o.keys():
                A[objects[o["pGroupSID"]]["node_id"],o["node_id"]] = 1
            for a_oid in o["aces"]:
                try:
                    A[objects[a_oid]["node_id"],o["node_id"]] = 1
                except KeyError as e:
                    pass
                    print(f"skipping: {e}")
        else:
            for a_oid in o["aces"]:
                try:
                    A[o["node_id"], objects[a_oid]["node_id"]] = 1
                except KeyError as e:
                    pass
                    print(f"skipping: {e}")
                         
        if np.sum(A[:,o["node_id"]]) == 0 and o["objectType"] != "domain":
            for key,value in objects.items():
                if value["dn"] == o["parent"]:
                    A[objects[key]["node_id"],o["node_id"]] = 1
                    break
            #groups, just connect them to the root
            if o["objectType"] == "group":
                A[0,o["node_id"]] = 1

    #delete loops
    for i in range(A.shape[0]):
        A[i,i] = 0
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    cycles = list(nx.simple_cycles(G))
    
    #break the cycle of the buil-in relations in administrators
    for c in cycles:
        A[c[0],c[1]] = 0
    #sort topologicaly
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    n_ordering = [k for k in nx.topological_sort(G)]
    new_A = transform_graph(nx.to_numpy_array(G), [k for k in nx.topological_sort(G)])
    X = X[n_ordering]
    return {"X":X, "A":new_A, "node_id_to_oid":node_id_to_oid, "objects":objects}


def preprocess_sharphound():
    
    classes = {"domain":0, "ou":1, "container":2, "group":3, "user":4, "computer":5}
    parser = ArgumentParser()
    parser.add_argument("--u_file", default="./users.json", type=str, help="Users JSON")
    parser.add_argument("--c_file", default="./computers.json", type=str, help="Computers JSON")
    parser.add_argument("--d_file", default="./domains.json", type=str, help="Domain JSON")
    parser.add_argument("--g_file", default="./groups.json", type=str, help="Groups JSON")
    parser.add_argument("--o_file", default="./ous.json", type=str, help="OUs JSON")
    parser.add_argument("--gp_file", default="./gpos.json", type=str, help="GPO JSON")
    parser.add_argument("--out", default="preprocessed_stratotest_domain.pickle", type=str, help="output_filename")
    args = parser.parse_args()

    data = process_sharphound_data(args)
    print(data)
    with open(args.out,"wb") as out_file:
        pickle.dump(data, out_file)
    return data


def add_connections(original_data, generated_data,user_data_csv,output_file="Enriched.json"):
    names = []
    with open(user_data_csv, "r") as csvfile:
        name_list = csv.reader(csvfile, delimiter=',')
        for line in name_list:
            names.append((line[1],line[2], f"{line[1].lower()}.{line[2].lower()}@rosta.com", f"{line[1].lower()}.{line[2].lower()}", line[4]))
    names = names[1:]
    output = []
    for i in range(generated_data.shape[0]):
        parents_ous = set()
        parents = []
        ou = None
        for j in range(generated_data.shape[1]):
            if(generated_data[i][j] > 0.1):
                if  original_data['objects'][original_data['node_id_to_oid'][j]]["objectType"]== 'user':
                    parents_ous.add(original_data['objects'][original_data['node_id_to_oid'][j]]["parent"])
                else:
                   parents.append(original_data['objects'][original_data['node_id_to_oid'][j]]["dn"])
                   if original_data['objects'][original_data['node_id_to_oid'][j]]["objectType"] == "ou":
                    ou = original_data['objects'][original_data['node_id_to_oid'][j]]["dn"]
        if not ou:
            ou = random.sample(parents_ous,1)
        output.append({"name":names[i][0], "surname":names[i][1], "email":names[i][2], "login":names[i][3], "password":names[i][4], "ou": ou, "parents": parents})
    with open("fake_users_stratodomain_random.json", 'w',encoding="utf-8") as fp:
        json.dump(output, fp)


def generate_random_honeyusers(A, num_nodes):
    mean = np.mean(np.sum(A, axis=1))
    ids = [i for i in range(A.shape[0])]
    new_nodes = np.zeros([num_nodes, A.shape[0]])
    for i in range(num_nodes):
        n = np.random.poisson(mean)
        for k in np.random.choice(ids,n,replace=False):
            new_nodes[i,k] = 1
        print(n, new_nodes[i,:])
    return new_nodes


# def create_honeyusers(node_id_to_oid, objects, parent_nodes):
#     """dsadd user <UserDN> [-samid <SAMName>] [-upn <UPN>] [-fn <FirstName>] [-mi <Initial>] [-ln <LastName>]
#         [-display <DisplayName>] [-empid <EmployeeID>] [-pwd {<Password> | *}] [-desc <Description>] [-memberof <Group> ...]
#         [-office <Office>] [-tel <PhoneNumber>] [-email <Email>] [-hometel <HomePhoneNumber>] [-pager <PagerNumber>] [-mobile <CellPhoneNumber>]
#         [-fax <FaxNumber>] [-iptel <IPPhoneNumber>] [-webpg <WebPage>] [-title <Title>] [-dept <Department>] [-company <Company>] [-mgr <Manager>]
#         [-hmdir <HomeDirectory>] [-hmdrv <DriveLetter>:][-profile <ProfilePath>] [-loscr <ScriptPath>] [-mustchpwd {yes | no}] [-canchpwd {yes | no}]
#         [-reversiblepwd {yes | no}] [-pwdneverexpires {yes | no}] [-acctexpires <NumberOfDays>] [-disabled {yes | no}] [{-s <Server> | -d <Domain>}]
#         [-u <UserName>] [-p {<Password> | *}] [-q] [{-uc | -uco | -uci}]
#     """
#     #generate the attributes of the user
#     unique_id = None
#     password = "superduper password"
#     upn = "user"
#     display  = "User Usersovic"
#     cn = ""
#     #contextual attributes
#     groups_dn = []
#     ou_dn = None
#     for n in parent_nodes:
#         if objects[node_id_to_oid[n]]["objectType"] == "group":
#             groups_dn.append(objects[node_id_to_oid[n]]["dn"])
#         elif objects[node_id_to_oid[n]]["objectType"] == "ou":
#             ou_dn = objects[node_id_to_oid[n]]["dn"]
#     #check that we have a OU
#     if not ou_dn:
#         #find default User OU (always present in AD) and place the user there
#         for o in objects.values():
#             if o["objectType"] == "ou" and o["rdn"].split("=")[-1] == "Users":
#                 ou_dn = o["dn"]         
#     user_dn = ",".join([cn,ou_dn])
    
#     #build the group dn sequence
#     groups = ''.join([f'"{x}"' for x in groups_dn])
#     #build the command
#     cmd = f'dsadd user {user_dn}' \
#         f'-samid "{unique_id}"' \
#         f'-upn "{upn}' \
#         f'-display "{display}"' \
#         '-disabled no' \
#         f'-pwd "{password}"' \
#         '-pwdneverexpires yes' \
#         '-accexpires never' \
#         f'-memberof {groups}' \

#     print(cmd)

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        original_data = pickle.load(f)
    predicted = generate_random_honeyusers(original_data["A"], 20)
    add_connections(original_data,predicted,"./names.csv","Enriched_stratodomain_random.json")
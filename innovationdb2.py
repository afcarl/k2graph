import sqlite3


#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.

# Written by Stefano Palmieri in February 2017

# This class stores unique genes in a database for the purpose of keeping track
# of innovation numbers. In this NEAT implementation, both node and link genes 
# are assigned innovation numbers. The sqlite rowid field is used as the 
# innovation number itself.


class InnovationDB2(object):
    def __init__(self):

        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()

        # rowid in the innovations table signifies
        # the innovation number for that gene
        self.cursor.execute('''CREATE TABLE link_innovations
                   (key INTEGER NOT NULL,
                   origin INTEGER NOT NULL,
                   destination INTEGER NOT NULL,
                   PRIMARY KEY (key, origin, destination));''')

        # rowid in the innovations table signifies
        # the innovation number for that gene
        self.cursor.execute('''CREATE TABLE node_innovations
                   (key INTEGER NOT NULL,
                   origin INTEGER NOT NULL,
                   PRIMARY KEY (key, origin));''')

    def close(self):
        self.conn.close()

    def retrieve_link_innovation_num(self, key, origin, destination):

        self.cursor.execute("SELECT rowid FROM link_innovations WHERE key=? AND origin=? AND destination=?",
                            (key, origin, destination))

        results = self.cursor.fetchall()

        # If the gene doesn't exist in the innovations table
        if not results:
            # add the gene to the innovation table and assign an innovation number
            self.cursor.execute("INSERT INTO link_innovations (key,origin,destination) VALUES (?, ?, ?)",
                                (key, origin, destination))
            return self.cursor.lastrowid
        else:
            for row in results:
                return row[0]

    def retrieve_node_innovation_num(self, key, origin):

        self.cursor.execute("SELECT rowid FROM node_innovations WHERE key=? AND origin=?",
                            (key, origin))

        results = self.cursor.fetchall()

        # If the gene doesn't exist in the innovations table
        if not results:
            # add the gene to the innovation table and assign an innovation number
            self.cursor.execute("INSERT INTO node_innovations (key,origin) VALUES (?, ?)",
                                (key, origin))
            return self.cursor.lastrowid
        else:
            for row in results:
                return row[0]


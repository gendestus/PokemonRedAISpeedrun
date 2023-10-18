import pymssql

class DB:
    def __init__(self, session_id = None):
        with open("dbinfo.txt", "r") as f:
            lines = f.readlines()
            self.server = lines[0].strip()
            self.user = lines[1].strip()
            self.password = lines[2].strip()
            self.database = lines[3].strip()

        if session_id is not None:
            self.session_id = session_id
        else:
            self.session_id = self.create_session()



        self.OAKLAB_TAG = "OAKLAB"
        self.PALLETTOWN_TAG = "PALLETTOWN"
        self.VIRIDIANCITY_TAG = "VIRIDIANCITY"
        self.CATCH_TAG = "CATCH"
        self.NEWMAP_TAG = "NEWMAP"
    
    def add_event(self, instanceID, description, eventTag):
        eventTypeID = self.get_event_type_id_by_tag(eventTag)
        with pymssql.connect(self.server, self.user, self.password, self.database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"EXEC AddEvent @instanceID = '{instanceID}', @description = '{description}', @eventType = '{eventTypeID}'")
                conn.commit()
    
    def add_gamestate(self, instanceID, gamestate: dict):
        enemyPokemonID = gamestate["enemyPokemonID"]
        enemyPokemonType1 = gamestate["enemyPokemonType1"]
        enemyPokemonType2 = gamestate["enemyPokemonType2"]
        enemyPokemonHealth = gamestate["enemyPokemonHealth"]
        mapID = gamestate["mapID"]  
        numBadges = gamestate["numBadges"]
        partyPokemonID1 = gamestate["partyPokemonID1"]
        partyPokemonID2 = gamestate["partyPokemonID2"]
        partyPokemonID3 = gamestate["partyPokemonID3"]
        partyPokemonID4 = gamestate["partyPokemonID4"]
        partyPokemonID5 = gamestate["partyPokemonID5"]
        partyPokemonID6 = gamestate["partyPokemonID6"]
        score = gamestate["score"]
        with pymssql.connect(self.server, self.user, self.password, self.database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"EXEC AddGamestate @instanceID = '{instanceID}', @enemyPokemonID = '{enemyPokemonID}', @enemyPokemonType1 = '{enemyPokemonType1}', @enemyPokemonType2 = '{enemyPokemonType2}', @enemyPokemonHealth = '{enemyPokemonHealth}', @mapID = '{mapID}', @numBadges = '{numBadges}', @partyPokemonID1 = '{partyPokemonID1}', @partyPokemonID2 = '{partyPokemonID2}', @partyPokemonID3 = '{partyPokemonID3}', @partyPokemonID4 = '{partyPokemonID4}', @partyPokemonID5 = '{partyPokemonID5}', @partyPokemonID6 = '{partyPokemonID6}', @score = '{score}'")
                conn.commit()

    def create_instance(self, sessionID) -> str:
        with pymssql.connect(self.server, self.user, self.password, self.database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"EXEC CreateInstance @sessionID = '{sessionID}'")
                instance_id = cursor.fetchone()[0]
                conn.commit()
                return str(instance_id)

    def create_session(self, notes = None) -> str:
        with pymssql.connect(self.server, self.user, self.password, self.database) as conn:
            with conn.cursor() as cursor:

                notes_sql = "NULL" if notes is None else f"'{notes}'"
                cursor.execute(f"EXEC CreateSession @notes = {notes_sql}")
                session_id = cursor.fetchone()[0]
                conn.commit()
                return str(session_id)

    def get_event_types(self) -> list:
        with pymssql.connect(self.server, self.user, self.password, self.database) as conn:
            with conn.cursor(as_dict=True) as cursor:
                cursor.execute("EXEC GetEventTypes")
                return cursor.fetchall()
    def get_event_type_id_by_tag(self, tag) -> str:
        event_types = self.get_event_types()
        for event_type in event_types:
            if event_type["Tag"] == tag:
                return event_type["TypeID"]
        return None
    
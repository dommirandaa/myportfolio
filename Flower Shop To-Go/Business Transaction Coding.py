import DBbase as db
import csv

class Flowers(db.DBbase):

    def __init__(self):
        super(Flowers, self).__init__("flowershoptogo.sqlite")

    def update(self, flower_type, flower_id):
        try:
            super().get_cursor.execute("update Flowers set flower_type = ? where id = ?;", (flower_id, flower_type))
            super().get_connection.commit()
            print(f"Updated record to {flower_type} successfully.")
        except Exception as e:
            print("An error has occurred", e)

    def add(self, flower_type):
        try:
            super().get_cursor.execute("insert or ignore into Flowers (flower_type) values(?);", (flower_type,))
            super().get_connection.commit()
            print(f"Add {flower_type} successfully.")
        except Exception as e:
            print("An error has occurred", e)
            return False

    def delete(self, flower_id):
        try:
            super().get_cursor.execute("DELETE FROM Flowers where id = ?", (flower_id,))
            super().get_connection.commit()
            print(f"Deleted flower id {flower_id} successfully.")
            return True
        except Exception as e:
            print("An error has occurred", e)

    def fetch(self, id=None, flower_name=None):
        try:
            if id is not None:
                return super().get_cursor.execute("SELECT * FROM Flowers WHERE id = ?", (id,)).fetchone()
            elif flower_name is not None:
                return super().get_cursor.execute("SELECT * FROM Flowers WHERE flower_type = ?", (flower_name,)).fetchone()
            else:
                return super().get_cursor.execute("SELECT * FROM Flowers").fetchall()
        except Exception as e:
            print("An error has occurred", e)

    def reset_database(self):
        try:
            sql = """
                DROP TABLE IF EXISTS Flowers;

                CREATE TABLE Flowers(
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                    flower_type TEXT UNIQUE );
            """
            super().execute_script(sql)
        except Exception as e:
            print("An error occurred", e)
        finally:
            super().close_db()


class Bouquet(Flowers):

    def add_bq(self, flower_type, qty, price):
        try:
            super().add(flower_type)
        except Exception as e:
            print("An error occurred in the flowers class", e)
        else:
            try:
                flower_id = super().fetch(flower_name=flower_type)[0]
                if flower_id is not None:
                    super().get_cursor.execute("""INSERT INTO Bouquet (flower_id, quantity, price) VALUES (?,?,?);""",(flower_id, qty, price))
                    super().get_connection.commit()
                    print(f"Bouquet {flower_type} added successfully")
                else:
                    raise Exception("The id of the flower name was not found.")
            except Exception as ex:
                print("An error occurred in the bouquet class.", ex)

    def update_bq(self, id, qty, price):
        try:
            super().get_cursor.execute("""UPDATE Bouquet SET quantity = ?, price = ? WHERE id = ?;""",(qty, price, id))
            super().get_connection.commit()
            print(f"Updated bouquet record successfully")
            return True
        except Exception as e:
            print("An error has occurred", e)
            return False

    def delete_bq(self, bouquet_id):
        try:
            flower_type = self.fetch_bq(bouquet_id)[1]
            if flower_type is not None:
                rsts = super().delete(flower_type)
                if rsts is False:
                    raise Exception("delete method in flowers failed. Delete aborted")
        except Exception as e:
            print("An error has occurred", e)
        else:
            try:
                super().get_cursor.execute("""DELETE FROM Bouquet where id = ?;""", (bouquet_id,))
            except:
                print("An error occurred in bouquet delete", e)

    def fetch_bq(self, id=None):
        try:
            if id is not None:
                retval = super().get_cursor.execute(
                    """SELECT Bouquet.id, flower_id, f.flower_type, quantity, price FROM Bouquet JOIN Flowers f on Bouquet.flower_id = f.id WHERE Bouquet.id = ?""",
                    (id,)).fetchone()
                return retval
            else:
                return super().get_cursor.execute(
                    """SELECT Bouquet.id, flower_id, f.flower_type, quantity, price FROM Flowers JOIN Flowers f on Bouquet.flower_id = f.id;""").fetchall()
        except Exception as e:
            print("An error has occurred", e)

    def reset_database(self):
        try:
            sql = """
                DROP TABLE IF EXISTS Bouquet;

                CREATE TABLE Bouquet(
                id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
                flower_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                price varchar(20)
                );
            """
            super().execute_script(sql)
            print("Bouquet table successfully created.")
        except Exception as e:
            print("An error has occurred", e)
        finally:
            super().close_db()

class Orders:

    def __init__(self,row):
        self.flower_id = row[0]
        self.flower_type = row[1]
        self.price = row[2]
        self.quantity = row[3]

class MoreOrders(db.DBbase):

#reset_or_create_db’ function coded to add ‘orders’ table with Orders fields
    def reset_or_create_db(self):

        try:
            sql = """
                DROP TABLE IF EXISTS Orders;

                CREATE TABLE Orders(
                    flower_id INTEGER NOT NULL,
                    flower_type TEXT UNIQUE,
                    price varchar(20),
                    quantity INTEGER NOT NULL
                );
            """
            super().execute_script(sql)

        except Exception as e:
            print(e)

    def read_order_data(self,flowershopdata):
        self.orders_list = []

        try:
            with open(flowershopdata,"r") as order:
                csv_content = csv.reader(order)
                next(order)
                for row in csv_content:
                    # print(row)
                    orders = Orders(row)
                    self.orders_list.append(orders)

        except Exception as e:
            print(e)

#‘save_to_database’ function coded to insert any information from csv
    def save_to_database(self):
        print("Number of orders to save: ",len(self.orders_list))
        save = input("Continue?: ").lower()

        if save == "y":
            for item in self.orders_list:
                item.flower_id = item.flower_id.replace("NaN","0")
                item.flower_type = item.flower_type.replace("NaN", "0")
                item.price = item.price.replace("NaN", "0")
                item.quantity = item.quantity.replace("NaN", "0")

                try:
                    super().get_cursor.execute(""" INSERT INTO Orders (flower_id,flower_type,price,quantity) VALUES(?,?,?,?)""",(item.flower_id,item.flower_type,item.price,item.quantity))
                    super().get_connection.commit()
                    print("Saved item: ",item.flower_id,item.flower_type,item.price)
                except Exception as e:
                    print(e)
        else:
            print("Save to DB aborted.")

#Last four lines of code reads and imports the data from ‘flowershopdata.csv’
orders = MoreOrders("flowershoptogo.sqlite")
orders.reset_or_create_db()
orders.read_order_data("flowershopdata.csv")
orders.save_to_database()

class Shop:

    def run(self):

        bq_option = {"get": "Get all bouquet.",
                      "getby": "Get bouquet by ID.",
                      "update": "Update Bouquet",
                      "add": "Add flowers",
                      "delete": "Delete Bouquet",
                      "reset": "Reset database.",
                      "exit": "Exit program"
                      }

        #The welcome phrase is printed to show the user the beginning of the interactive menu.
        print("Welcome to my Shop, please choose a selection of flowers.")

        user_selection = ''
        while user_selection != "exit":
            print("*** Flowers Options List ***")
            for option in bq_option.items():
                print(option)

            user_selection = input("Select an option: ") #Each option is printed with a required input
            bouquet = Bouquet()

            if user_selection == "get":
                results = bouquet.fetch_bq()
                for item in results:
                    print(item)

            elif user_selection == "getby":

                bq_id = input("Enter Bouquet Id: ")
                results = bouquet.fetch_bq(bq_id)
                print(results)
                input("Press return to continue")

            elif user_selection == "update":
                results = bouquet.fetch_bq()
                for item in results:
                    print(item)

                bq_id = input("Enter Bouquet Id: ")
                qty = input("Enter quantity amount: ")
                price = input("Enter unit price: ")
                bouquet.update_bq(bq_id , qty, price)
                input("Press return to continue")

            elif user_selection == "add":
                flower_name = input("Enter flower name: ")
                qty = input("Enter quantity amount: ")
                price = input("Enter unit price: ")
                bouquet.add_bq(flower_name, qty, price)
                print("Done\n")
                input("Press return to continue")

            elif user_selection == "delete":
                bq_id = input("Enter Bouquet Id: ")
                bouquet.delete_bq(bq_id)
                print("Done\n")
                input("Press return to continue")

            elif user_selection == "reset":
                confirm = input("This will delete all records in parts and inventory, continue? (y/n) ").lower()
                if confirm == "y":
                    Bouquet.reset_database()
                    parts = Flowers()
                    parts.reset_database()
                    print("Reset complete")
                    input("Press return to continue")
                else:
                    print("Reset aborted.")
                    input("Press return to continue")
            else:
                if user_selection != "exit":
                    print("Invalid selection. Please try again\n") #menu has prompts for user to input information as directed


# shop = Shop()
# shop.run()

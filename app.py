from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from pulp import *
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.patches as mpatches

def create_app():
    app = Flask(__name__)
    app.secret_key = "hello"
    app.permanent_session_lifetime = timedelta(minutes=5)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scheduling_data_1.sqlite3'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db = SQLAlchemy(app)

    class Patient(db.Model):
        _id = db.Column(db.Integer, primary_key=True)
        pat_name = db.Column(db.String(50), nullable=False)
        treat_hrs_1st_appointment = db.Column(db.Integer)
        treat_type_1st_appointment = db.Column(db.String(50))
        treat_hrs_2nd_appointment = db.Column(db.Integer)
        treat_type_2nd_appointment = db.Column(db.String(50))
        treat_hrs_3rd_appointment = db.Column(db.Integer)
        treat_type_3rd_appointment = db.Column(db.String(50))

        def __init__(self, pat_name, treat_hrs_1st_appointment=None, treat_type_1st_appointment=None,
                     treat_hrs_2nd_appointment=None, treat_type_2nd_appointment=None,
                     treat_hrs_3rd_appointment=None, treat_type_3rd_appointment=None):
            self.pat_name = pat_name
            self.treat_hrs_1st_appointment = treat_hrs_1st_appointment
            self.treat_type_1st_appointment = treat_type_1st_appointment
            self.treat_hrs_2nd_appointment = treat_hrs_2nd_appointment
            self.treat_type_2nd_appointment = treat_type_2nd_appointment
            self.treat_hrs_3rd_appointment = treat_hrs_3rd_appointment
            self.treat_type_3rd_appointment = treat_type_3rd_appointment

    # Function to add or update patient data
    def add_or_update_patient_data(pat_name, treat_hrs, treat_type, appoint_order):
        try:
            # Check if the patient already exists
            patient = Patient.query.filter_by(pat_name=pat_name).first()

            # If the patient exists, update the data for the specified appointment order
            if patient:
                setattr(patient, f"treat_hrs_{appoint_order.lower().split()[0]}_appointment", treat_hrs)
                setattr(patient, f"treat_type_{appoint_order.lower().split()[0]}_appointment", treat_type)
            else:
                # If the patient does not exist, create a new patient entry
                new_patient = Patient(
                    pat_name=pat_name,
                    treat_hrs_1st_appointment=None,
                    treat_type_1st_appointment=None,
                    treat_hrs_2nd_appointment=None,
                    treat_type_2nd_appointment=None,
                    treat_hrs_3rd_appointment=None,
                    treat_type_3rd_appointment=None
                )
                setattr(new_patient, f"treat_hrs_{appoint_order.lower().split()[0]}_appointment", treat_hrs)
                setattr(new_patient, f"treat_type_{appoint_order.lower().split()[0]}_appointment", treat_type)
                db.session.add(new_patient)

            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error adding/updating patient data: {str(e)}")
            return False
        finally:
            db.session.close()


    # Function to clear data from the Patient table
    def clear_patient_data():
        try:
            Patient.query.delete()
            db.session.commit()
            print("Patient data cleared successfully.")
        except Exception as e:
            db.session.rollback()
            print(f"Error clearing patient data: {str(e)}")
        finally:
            db.session.close()

    # Function to get data from the database
    def get_data_from_database():
        with app.app_context():
            try:
                # Query the database to get all patient records
                patients = Patient.query.all()

                # Initialize lists for visit durations and clinic sequences
                times = []
                clinic_sequence = []

                # Map treatment types to their corresponding codes
                treat_type_mapping = {"Clinic Visit": 0, "Infusion": 1, "Nurse Follow-Up": 2}

                # Iterate over each patient record
                for patient in patients:
                    # Create a list to store visit durations for the current patient
                    durations = [None, None, None]
                    seq = []

                    # Extract treatment hours and treat types for each appointment order
                    for order in ["1st Appointment", "2nd Appointment", "3rd Appointment"]:
                        treat_type = getattr(patient, f"treat_type_{order.lower().split()[0]}_appointment")
                        treat_hours = getattr(patient, f"treat_hrs_{order.lower().split()[0]}_appointment")

                        # Convert treat type to the corresponding code
                        treat_code = treat_type_mapping.get(treat_type, None)
                        # print(treat_code)

                        # Append the visit duration to the list
                        if treat_code is not None:
                            durations[treat_code] = treat_hours

                        # If the treat type is not None, append the visit code to clinic_sequence
                        if treat_code is not None:
                            seq.append(treat_code)

                    # Append the list of visit durations for the current patient to times
                    times.append(durations)
                    clinic_sequence.append(seq)

                return times, clinic_sequence
            except Exception as e:
                print(f"Error getting data from the database: {str(e)}")
                return None, None

    # ... (your data retrieval code)

    # Function for optimization and plotting
    def optimize_and_plot(times, clinic_sequence):

        # Optimization Algorithm
        # Convenience Variables
        patients = len(clinic_sequence)
        valid_starts = [(j, m) for j in range(patients) for m in clinic_sequence[j]]
        valid_starts.sort()

        jjm = [(j, j_prime, m) for j in range(patients) for j_prime in range(patients) for m in
               range(len(clinic_sequence[j]))
               if j != j_prime and (j, m) in valid_starts and (j_prime, m) in valid_starts]

        # Objective
        schedule = LpProblem(name='Minimize_Schedule', sense=LpMinimize)

        # Variables
        x = LpVariable.dicts("start_time", indices=valid_starts, lowBound=0, cat='Continuous')
        y = LpVariable.dicts("precedence", indices=jjm, cat='Binary')
        M = 100  # A big M constant to ensure the correctness of the constraint
        c = LpVariable('makespan', lowBound=0, cat='Continuous')

        # Constraints
        # Visit sequence constraint
        for j in range(patients):
            for m_idx in range(1, len(clinic_sequence[j])):
                curr_visit = clinic_sequence[j][m_idx]
                prior_visit = clinic_sequence[j][m_idx - 1]
                schedule += x[j, curr_visit] >= x[j, prior_visit] + times[j][prior_visit]

        # Single-use constraint
        for j, j_prime, m in jjm:
            schedule += x[j, m] >= x[j_prime, m] + times[j_prime][m] - y[j, j_prime, m] * M
            schedule += x[j_prime, m] >= x[j, m] + times[j][m] - (1 - y[j, j_prime, m]) * M

        # Makespan constraint
        schedule += c
        for j, m in valid_starts:
            schedule += c >= x[j, m] + times[j][m]

        # Solve the model
        status = schedule.solve()

        # Show the Results
        # print(f"status: {schedule.status}, {LpStatus[schedule.status]}")
        # print(f"Completion Time: {schedule.objective.value()}")

        # for j, m in valid_starts:
        #     if x[j, m].varValue >= 0:
        #         print(f"Patient {j + 1} starts in clinic {m} at time {x[j, m].varValue}")

        # Create a full list of start times
        start_times = []
        for j, m in valid_starts:
            if x[j, m].varValue >= 0:
                start_times.append(x[j, m].varValue)

        # create full list of duration
        duration = []
        for time in times:
            for i in time:
                duration.append(i)
        duration = [x for x in duration if x is not None]

        # create a full list of visits
        visits = []
        for j, m in valid_starts:
            if x[j, m].varValue >= 0:
                visits.append(m)

        # create a full list of patients
        patients = []
        for j, m in valid_starts:
            if x[j, m].varValue >= 0:
                patients.append(j)

        # create the indexes for all visits
        machine_1 = [i for i, e in enumerate(visits) if e == 0]
        machine_2 = [i for i, e in enumerate(visits) if e == 1]
        machine_3 = [i for i, e in enumerate(visits) if e == 2]
        # machine_1, machine_2, machine_3

        # Set up matplotlib figure and axis
        fig, gnt = plt.subplots()

        # Set the axes names
        gnt.set_xlabel("Time (in hours)")
        gnt.set_ylabel("Clinic Visits")

        # Set axes limits
        gnt.set_ylim(9, 40)
        gnt.set_xlim(0, schedule.objective.value())

        # Set the ticks on the x-axis
        gnt.set_xticks(range(0, int(schedule.objective.value()) + 1))  # Adjusted this line

        # Set y-axis tick labels
        gnt.set_yticklabels(['Clinic Visit', 'Infusion', 'Nurse Follow-Up'])
        gnt.set_yticks([15, 25, 35])

        # Set graph attributes
        gnt.grid(True)

        # Define colors for each patient
        colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:red']

        # Define a mapping of patient indices to patient names
        patient_names = {0: 'Patient A', 1: 'Patient B', 2: 'Patient C', 3: 'Patient D', 4: 'Patient E'}

        # Create legend patches
        legend_patches = [mpatches.Patch(color=color, label=patient_names[patient]) for patient, color in
                          enumerate(colors)]

        # Plot schedules for each machine
        def plot_schedule(machine, y_coord, colors):
            ncolor = 0
            for idx, val in enumerate(machine):
                try:
                    gnt.broken_barh([(start_times[val], duration[val])],  # Adjusted this line
                                    (y_coord, 9),
                                    facecolors=colors[ncolor])
                    ncolor += 1
                except IndexError:
                    pass

        # Plot schedules for each machine
        plot_schedule(machine_1, 10, colors)
        plot_schedule(machine_2, 20, colors)
        plot_schedule(machine_3, 30, colors)

        # Add legend outside the plot
        plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

        plt.grid(True)

        # Convert the Matplotlib plot to a base64-encoded image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
        plt.close()  # Close the plot to prevent it from being displayed twice

        # Return the base64-encoded image data
        return img_base64

    # Flask route to display the result
    @app.route('/result', methods=['GET'])
    def show_result():
        # Get data from the database
        times, clinic_sequence = get_data_from_database()

        # Optimize and plot
        img_base64 = optimize_and_plot(times, clinic_sequence)

        if img_base64:
            # Render the HTML page with the base64-encoded image
            return render_template('result.html', img_base64=img_base64)
        else:
            return "Error generating optimization result image."

    @app.route('/')
    def home():
        return redirect(url_for("add_patient"))

    # Route to clear patient data
    @app.route("/clear_patient_data")
    def clear_patient_data_route():
        clear_patient_data()
        return "Patient data cleared successfully!"

    # Route to add or update patient data
    @app.route('/add_patient', methods=['POST', 'GET'])
    def add_patient():
        if request.method == "POST":
            pat_name = request.form["patientName"]
            session["name"] = pat_name
            treat_hrs = request.form["treatmentHours"]
            treat_type = request.form["treatmentType"]
            appoint_order = request.form["appointmentOrder"]

            # Call the function to add or update patient data
            success = add_or_update_patient_data(pat_name, treat_hrs, treat_type, appoint_order)

            if success:
                flash(f"Patient data for {pat_name} ({appoint_order}) added/updated successfully!")
            else:
                flash("Error adding/updating patient data. Please try again.")

            # Retrieve all patient data from the database
            all_patient_data = Patient.query.all()

            # print(all_patient_data)

            return render_template('result.html', all_patient_data=all_patient_data)
        else:
            return render_template('index.html')

    return app, db, Patient


def init_db(db):
    with app.app_context():
        db.create_all()


if __name__ == '__main__':
    app, db, _ = create_app()
    init_db(db)
    app.run(debug=True)



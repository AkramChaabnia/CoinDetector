import requests
import base64


def retrieve_heart_rate_data():
    # Hardcoded credentials (replace with your actual values)
    client_id = "c01de748-09ab-46bf-b63a-642fe4a81431"
    client_secret = "fb27b5cd-4a6b-41f7-8bc9-620d62309ee2"
    date = "2023-08-24"  # Specific date to retrieve data for (optional)

    # Step 1: Build authorization URL
    authorization_url = "https://flow.polar.com/oauth2/authorization"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": "accesslink.read_all"  # Request accesslink.read_all scope for user data
    }
    authorization_url += "?" + "&".join([f"{key}={value}" for key, value in params.items()])

    # Inform the user about redirection and potential interaction with Polar
    print(f"Authorization URL: {authorization_url}")  # Added print statement

    # User interaction to obtain authorization code (not implemented here)
    # ... (refer to previous explanation for steps 3 and potentially modify your app)

    # Placeholder for user-obtained authorization code
    authorization_code = input("Enter the authorization code obtained from Polar: ")

    # Step 4: Build token request URL
    token_url = "https://polarremote.com/v2/oauth2/token"

    # Step 5: Encode client credentials (base64)
    combined = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(combined.encode("utf-8")).decode("utf-8")
    authorization_header = f"Basic {encoded_credentials}"

    # Step 6: Prepare request body
    data = {
        "grant_type": "authorization_code",
        "code": authorization_code,
    }

    # Step 7: Send token request
    headers = {"Authorization": authorization_header, "Content-Type": "application/x-www-form-urlencoded"}
    print(f"Sending token request to: {token_url}")  # Added print statement
    token_response = requests.post(token_url, headers=headers, data=data)

    # Step 8: Handle token response
    if token_response.status_code == 200:
        access_token = token_response.json()["access_token"]
        user_id = token_response.json().get("x_user_id")  # Extract user ID from response (if available)
        #print the token response
        print("Token response: ", token_response.json()
        # Optional user registration (uncomment if needed)
        user_registration_url = "https://www.polaraccesslink.com/v3/users"
        #user_id = "your_unique_user_id"  # Replace with your identifier for the user
        registration_data = {"member-id": user_id}
        registration_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/xml",
            "Accept": "application/json",
        }
        registration_response = requests.post(user_registration_url, headers=registration_headers, json=registration_data)
        if registration_response.status_code == 200:
            print(f"User with ID '{user_id}' successfully registered.")
        else:
            print(f"User registration failed: {registration_response.text}")

        # ... (optional code for heart rate data retrieval)

        return access_token, user_id  # Return both access token and user ID
    else:
        raise Exception(f"Failed to retrieve access token: {token_response.text}")


def retrieve_user_information(access_token, user_id):
    # URL to retrieve user information
    user_info_url = f"https://www.polaraccesslink.com/v3/users/{user_id}"

    # Set headers with access token
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    # Send GET request to retrieve user information
    user_response = requests.get(user_info_url, headers=headers)

    # Handle response
    if user_response.status_code == 200:
        return user_response.json()
    else:
        raise Exception(f"Failed to retrieve user information: {user_response.text}")


def main():
    # Testing the function
    try:
        access_token, user_id = retrieve_heart_rate_data()
        print(f"Retrieved access token: {access_token}")
        print(f"Retrieved user ID: {user_id}")  # Print the retrieved user ID

        # Retrieve user information using the access token
        user_information = retrieve_user_information(access_token, user_id)
        print("User Information:")
        # Print specific user information fields (optional)
        # for key, value in user_information.items():
        #     print(f"{key}: {value}")
        print(user_information)  # Print the entire user information dictionary

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

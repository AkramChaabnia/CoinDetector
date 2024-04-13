import requests
import base64


def get_authorization_url(client_id, scope):
    """
    Builds the authorization URL for Polar OAuth2 flow.

    This function takes your client ID and the desired access scope (e.g., "accesslink.read_all") as arguments and constructs the complete authorization URL. The URL directs users to Polar's login page where they can grant access to your application.

    Args:
        client_id (str): Your Polar API client ID.
        scope (str): The scope requested (e.g., "accesslink.read_all").

    Returns:
        str: The authorization URL.
    """
    base_url = "https://flow.polar.com/oauth2/authorization"
    params = {"response_type": "code", "client_id": client_id, "scope": scope}
    return base_url + "?" + "&".join([f"{key}={value}" for key, value in params.items()])


def handle_reauthorization(access_token, user_id):
    """
    Handles reauthorization flow after a 403 error (Forbidden).

    This function informs the user about missing consents, prompts them to reauthorize, and obtains a new access token.

    Args:
        access_token (str): The current (potentially invalid) access token.
        user_id (str): The user ID.

    Returns:
        dict or None: A dictionary containing user information on success (after reauthorization), None on error.
    """
    print("User might be missing required consents. Please reauthorize the application in Polar settings:")
    print(f"Polar Account Settings: https://account.polar.com/")

    # User interaction to reauthorize (outside this code example)
    # ... (consider using input() to prompt the user to continue)

    # Placeholder for user to confirm reauthorization
    user_confirmed_reauthorization = True  # Replace with actual user confirmation logic

    if user_confirmed_reauthorization:
        # Repeat the authorization flow to get a new access token
        # (consider using get_access_token function again)
        new_access_token, _ = get_access_token(client_id, client_secret, authorization_code)  # Placeholder for new code
        print("Reauthorization successful. Retrieving user information again...")
        return get_user_information(new_access_token, user_id)  # Call get_user_information with new token
    else:
        print("User did not confirm reauthorization. Exiting.")
        return None

def get_access_token(client_id, client_secret, authorization_code):
    """
    Retrieves access token and user ID from Polar using the authorization code.

    This function handles the exchange of the authorization code obtained from Polar's login flow for an access token. It also retrieves the user ID associated with the granted access.

    Args:
        client_id (str): Your Polar API client ID.
        client_secret (str): Your Polar API client secret.
        authorization_code (str): The authorization code obtained from Polar.

    Returns:
        tuple: A tuple containing the access token (str) and user ID (str or None).
    """
    token_url = "https://polarremote.com/v2/oauth2/token"
    encoded_credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    authorization_header = f"Basic {encoded_credentials}"

    data = {"grant_type": "authorization_code", "code": authorization_code}
    headers = {"Authorization": authorization_header, "Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()  # Raise exception for non-200 status codes

        access_token = response.json()["access_token"]
        user_id = response.json().get("x_user_id")
        return access_token, user_id
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving access token: {e}")
        raise


def register_user(access_token, member_id):
    """
    Registers a user with Polar using the access token and a custom member ID.

    This function performs the user registration step as mandated by the Polar AccessLink API. It requires a valid access token and your custom identifier for the user within your application.

    Args:
        access_token (str): The access token obtained from Polar.
        member_id (str): Your custom identifier for the user.

    Returns:
        None
    """
    url = "https://www.polaraccesslink.com/v3/users"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/xml", "Accept": "application/json"}
    data = {"member-id": member_id}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for non-200 status codes

        if response.status_code == 200:
            print(f"User with ID '{member_id}' successfully registered.")
        else:
            print(f"User registration failed: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error during user registration: {e}")

        
def handle_reauthorization(access_token, user_id):
    """
    Handles reauthorization flow after a 403 error (Forbidden).

    This function informs the user about missing consents, prompts them to reauthorize, and obtains a new access token.

    Args:
        access_token (str): The current (potentially invalid) access token.
        user_id (str): The user ID.

    Returns:
        dict or None: A dictionary containing user information on success (after reauthorization), None on error.
    """
    print("User might be missing required consents. Please reauthorize the application in Polar settings:")
    print(f"Polar Account Settings: https://account.polar.com/")

    # User interaction to reauthorize (outside this code example)
    # ... (consider using input() to prompt the user to continue)

    # Placeholder for user to confirm reauthorization
    user_confirmed_reauthorization = True  # Replace with actual user confirmation logic

    if user_confirmed_reauthorization:
        # Repeat the authorization flow to get a new access token
        # (consider using get_access_token function again)
        new_access_token, _ = get_access_token(client_id, client_secret, authorization_code)  # Placeholder for new code
        print("Reauthorization successful. Retrieving user information again...")
        return get_user_information(new_access_token, user_id)  # Call get_user_information with new token
    else:
        print("User did not confirm reauthorization. Exiting.")
        return None



def get_user_information(access_token, user_id):
    """
    Retrieves user information from Polar using the access token and user ID.

    This function retrieves user data associated with the provided user ID. It requires a valid access token for authorization.

    Args:
        access_token (str): The access token obtained from Polar.
        user_id (str): The user ID retrieved from Polar.

    Returns:
        dict or None: A dictionary containing user information on success, None on error.
    """
    url = f"https://www.polaraccesslink.com/v3/users/{user_id}"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for non-200 status codes

        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("Access denied. Please check your access token and permissions.")
            return handle_reauthorization(access_token, user_id)
        else:
            print(f"Error retrieving user information: {e}")
            return None


def retrieve_heart_rate_data(access_token, user_id, date):
    """
    Retrieves heart rate data (optional) for a specific date using Polar API.

    This function retrieves heart rate data for a user (assuming you have the specific endpoint URL). It requires your client ID, client secret, and an optional date parameter (YYYY-MM-DD format).

    Args:
        client_id (str): Your Polar API client ID.
        client_secret (str): Your Polar API client secret.
        date (str, optional): The date for which to retrieve heart rate data (YYYY-MM-DD). Defaults to None.

    Returns:
        dict or None: A dictionary containing heart rate data on success, None on error.
    """

    # Replace with your actual Polar API endpoint for heart rate data retrieval
    heart_rate_url = f"https://www.polaraccesslink.com/v3/users/continuous-heart-rate/{date}"
    headers = {"Authorization": f"Bearer {access_token}"}

    #register_user(access_token, f"user_id_{user_id}")  # Example custom member ID

    try:
        response = requests.get(heart_rate_url, headers=headers)
        response.raise_for_status()  # Raise exception for non-200 status codes

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving heart rate data: {e}")
        return None


if __name__ == "__main__":
    # Replace with your actual Polar API credentials
    client_id = "c01de748-09ab-46bf-b63a-642fe4a81431"
    client_secret = "fb27b5cd-4a6b-41f7-8bc9-620d62309ee2"

    # Inform the user about redirection and potential interaction with Polar
    print(f"Authorization URL: {get_authorization_url(client_id, 'accesslink.read_all')}")

    # Placeholder for user-obtained authorization code
    authorization_code = input("Enter the authorization code obtained from Polar: ")


    # Step 1: Retrieve Access Token and User ID
    try:
        access_token, user_id = get_access_token(client_id, client_secret, authorization_code)
        print("Access token retrieved successfully.")
        print(f"Retrieved access token: {access_token}")
        print(f"Retrieved user ID: {user_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving access token: {e}. Possibly due to an invalid authorization code.")
        exit(1)  # Exit the program with an error code

    # Step 2: User Registration (if required by Polar API)
    #member_id = f"user_id_{user_id}"  # Example custom member ID
    #register_user(access_token, member_id)

# Step 3: Retrieve User Information
try:
    user_information = get_user_information(access_token, user_id)
    if user_information is None:
        print("Failed to retrieve user information.")
    else:
        print("User Information:")
        # Print specific user information fields (optional)
        for key, value in user_information.items():
             print(f"{key}: {value}")
        print(user_information)  # Print the entire user information dictionary
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 403:
        print("Access denied. Please check your access token and permissions.")
        handle_reauthorization(access_token, user_id)
    else:
        print(f"Error retrieving user information: {e}")

# Step 4 (Optional): Retrieve Heart Rate Data (replace with your endpoint)
date = "2024-04-10"  # Example date (optional)
try:
    heart_rate_data = retrieve_heart_rate_data(access_token, user_id, date)
    if heart_rate_data:
        print("Heart rate data retrieved successfully!")
        # Process or display the retrieved heart rate data (heart_rate_data dictionary)
    else:
        print("An error occurred while retrieving heart rate data.")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 403:
        print("Access denied. Please check your access token and permissions.")
    else:
        print(f"Error retrieving heart rate data: {e}")
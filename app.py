```python
# ============================================================
# GOOGLE AUTHENTICATION
# ============================================================

def google_auth_available():
    """
    Check whether Streamlit Google OAuth is properly configured.
    Required secrets:

    [auth]
    redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
    cookie_secret = "your-random-secret"

    [auth.google]
    client_id = "your-client-id"
    client_secret = "your-client-secret"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
    """
    try:
        # Check [auth]
        if "auth" not in st.secrets:
            return False

        auth_config = st.secrets["auth"]

        # Check basic auth settings
        if not auth_config.get("redirect_uri"):
            return False

        if not auth_config.get("cookie_secret"):
            return False

        # Check Google settings
        if "google" not in auth_config:
            return False

        google_config = auth_config["google"]

        if not google_config.get("client_id"):
            return False

        if not google_config.get("client_secret"):
            return False

        if not google_config.get("server_metadata_url"):
            return False

        return True

    except Exception:
        return False


def handle_google_login():
    """Start Google OAuth login safely."""
    if not google_auth_available():
        st.error(
            "Google Sign-In is not configured correctly. "
            "Please check Streamlit Cloud → Settings → Secrets."
        )
        return

    try:
        st.login("google")
    except Exception as e:
        st.error(
            "Google Sign-In could not start. "
            "Please check your Google OAuth settings in Streamlit Cloud."
        )


def login_page():

    # --------------------------------------------------------
    # Check whether Google login has already succeeded
    # --------------------------------------------------------
    if google_auth_available():

        try:
            if getattr(st.user, "is_logged_in", False):

                st.session_state.logged_in = True

                st.session_state.user_name = (
                    getattr(st.user, "name", None)
                    or getattr(st.user, "email", "Google User")
                )

                st.session_state.user_email = (
                    getattr(st.user, "email", "")
                )

                st.session_state.page = "Home"

                st.rerun()

        except Exception:
            pass


    # --------------------------------------------------------
    # LOGIN UI
    # --------------------------------------------------------

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            '<div class="logo-box">🕵️</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<h2 class="hero-title" '
            'style="text-align:center; font-size:26px;">'
            'Fake Review Detector'
            '</h2>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<p style="text-align:center; color:#5A5A5A; '
            'font-size:13.5px; margin-bottom:20px;">'
            'Sign in to access the Fake Review Detection System.'
            '</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-box">',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # GOOGLE SIGN-IN
        # ----------------------------------------------------

        if google_auth_available():

            st.button(
                "🔵 Sign in with Google",
                on_click=handle_google_login,
                use_container_width=True,
                type="primary"
            )

            st.markdown(
                '<p style="text-align:center; color:#5A5A5A; '
                'font-size:12px; margin:14px 0;">'
                '— OR use the demo admin account below —'
                '</p>',
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "Google Sign-In is not configured. "
                "You can still use the demo admin account."
            )


        # ----------------------------------------------------
        # DEMO ADMIN LOGIN
        # ----------------------------------------------------

        username = st.text_input(
            "👤 Username",
            placeholder="Enter your username"
        )

        password = st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter your password"
        )


        if st.button(
            "LOGIN  →",
            type="secondary" if google_auth_available() else "primary",
            use_container_width=True
        ):

            if (
                username == VALID_USERNAME
                and password == VALID_PASSWORD
            ):

                st.session_state.logged_in = True
                st.session_state.user_name = "Admin (Demo)"
                st.session_state.user_email = "admin-demo@local"
                st.session_state.page = "Home"

                st.rerun()

            else:

                st.error(
                    "❌ Invalid username or password. Please try again."
                )


        # ----------------------------------------------------
        # DEMO CREDENTIALS
        # ----------------------------------------------------

        st.markdown(
            f'''
            <div class="hint-box">
            ℹ️ <b>Demo credentials</b> —
            Username: <b>{VALID_USERNAME}</b>
            &nbsp;|&nbsp;
            Password: <b>{VALID_PASSWORD}</b>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MAIN APP FLOW
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    # --------------------------------------------------------
    # SIDEBAR USER INFORMATION
    # --------------------------------------------------------

    st.sidebar.markdown(
        f"""
        👋 **{st.session_state.get('user_name', 'User')}**
        <br>
        <span style='font-size:12px; opacity:0.8;'>
        {st.session_state.get('user_email', '')}
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    st.sidebar.title("📂 Navigation")

    pages = [
        "Home",
        "Review Checker",
        "Batch Checker",
        "Insights",
        "Dashboard"
    ]

    nav_choice = st.sidebar.radio(
        "Go to:",
        pages,
        index=pages.index(st.session_state.page)
    )

    st.session_state.page = nav_choice


    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------

    if st.sidebar.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.page = "Home"

        # Logout from Google only when Google authentication
        # is actually configured and the user is logged in.
        if google_auth_available():

            try:

                if getattr(
                    st.user,
                    "is_logged_in",
                    False
                ):
                    st.logout()

            except Exception:
                pass

        st.rerun()


    # --------------------------------------------------------
    # PAGE ROUTING
    # --------------------------------------------------------

    if st.session_state.page == "Home":

        home_page()

    elif st.session_state.page == "Review Checker":

        review_checker_page()

    elif st.session_state.page == "Batch Checker":

        batch_checker_page()

    elif st.session_state.page == "Insights":

        insights_page()

    elif st.session_state.page == "Dashboard":

        dashboard_page()
```

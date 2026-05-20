from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import yt_dlp, os
import vlc
from logger import CustomLogger  # Import CustomLogger
from prompts.music_prompt import music_few_shot_examples, music_prompt_template


class MusicModule:
    """
    Handles music-related functionality, including extracting and playing music from YouTube.
    """

    def __init__(self, api_key):
        """
        Initialize the Music Module with the required API key.
        """
        # Initialize the Groq LLM for music query processing
        self.llm_music = ChatGroq(
            api_key=api_key,
            model="llama3-8b-8192",
            temperature=0.4
        )

        # Few-shot examples for music queries
        self.music_few_shot_examples = music_few_shot_examples

        # Initialize VLC player (global state for playing music)
        self.music_player = None

        # Initialize logger
        self.logger = CustomLogger().get_logger()
        self.logger.info("MusicModule initialized.")

    def refine_music_query(self, user_input):
        """
        Refines the user's music request using the LLM.
        :param user_input: User's raw input.
        :return: Refined query extracted by the LLM.
        """
        self.logger.info(f"Refining music query for input: {user_input}")
        few_shot_text = "\n".join(
            [f"Input: {ex['input']} Query: {ex['query']}" for ex in self.music_few_shot_examples]
        )
        prompt = music_prompt_template(user_input, few_shot_text)
        messages = [
            SystemMessage(content="You are a music query refinement assistant, specializing in accurate song identification."),
            HumanMessage(content=prompt)
        ]
        try:
            response = self.llm_music.invoke(messages)
            refined_query = response.content.strip()
            self.logger.info(f"Refined query: {refined_query}")
            return refined_query
        except Exception as e:
            self.logger.error(f"Error during query refinement: {e}")
            return None

    def fetch_and_play_music(self, query):
        """
        Searches for music on YouTube and streams it directly using VLC.
        """
        self.logger.info(f"Fetching and attempting to play music for query: {query}")
        try:
            # Configure yt-dlp to extract the streaming URL
            ydl_opts = {
                'format': 'bestaudio/best',
                'noplaylist': True,
                'quiet': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(f"ytsearch:{query}", download=False)
                if not results.get('entries'):
                    self.logger.warning(f"No results found for query: {query}")
                    return "No results found for your query. Please try a different song."

                # Extract the first result
                result = results['entries'][0]
                video_title = result['title']
                video_url = result['url']
                self.logger.info(f"Found music: {video_title} - URL: {video_url}")

                # Stop any currently playing music
                self.stop_music()

                # Add headers for VLC to handle YouTube URLs
                media = vlc.Media(video_url)
                media.add_options(
                    ":http-user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    ":http-referrer=https://www.youtube.com/",
                    ":network-caching=1000"
                )

                # Play the stream
                self.music_player = vlc.MediaPlayer()
                self.music_player.set_media(media)
                self.music_player.audio_set_volume(100)  # Set volume to 100%
                self.music_player.play()

                self.logger.info(f"Now playing: {video_title}")
                return f"Playing: {video_title}. Click 'Interrupt' or 'Exit' to stop the music."
        except Exception as e:
            self.logger.error(f"Error during music playback: {e}")
            return f"An error occurred while playing the song: {e}"

    def stop_music(self):
        """
        Stops the currently playing music, if any.
        """
        if self.music_player:
            self.logger.info("Stopping the currently playing music.")
            self.music_player.stop()
            self.music_player = None
        else:
            self.logger.info("No music is currently playing to stop.")

    def handle_play_music(self, user_input):
        """
        Handles user requests for music playback.
        :param user_input: User's raw input for playing music.
        :return: Response message.
        """
        self.logger.info(f"Handling music request: {user_input}")
        # Handle stop-related commands
        if user_input.lower().strip() in ["interrupt", "stop", "exit", "quit"]:
            self.stop_music()  # Stop the music playback

            if user_input.lower().strip() == "interrupt":
                self.logger.info("Music playback interrupted.")
                return " "
            elif user_input.lower().strip() == "exit":
                self.logger.info("Exited from the current functionality.")
                return "Exited from the current functionality."
            elif user_input.lower().strip() == "quit":
                self.logger.info("Music stopped. Redirecting to Groq API key page.")
                return "Music stopped. Redirecting to Groq API key page."

        # Refine the query for playing music
        refined_query = self.refine_music_query(user_input)
        if not refined_query:
            self.logger.warning("Refinement failed or no query extracted.")
            return "I couldn't understand the song you want to play. Could you try rephrasing?"

        # Stop any currently playing music before starting a new one
        self.stop_music()

        # Fetch and play the song
        return self.fetch_and_play_music(refined_query)

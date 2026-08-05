import "./Home.css";
import owlIcon from "../../assets/owl-icon.png";

export default function Home() {
    return (
        <main className="home">

            <div className="hero">

                <img src={owlIcon} alt="Owly logo" className="logo" />
                <h1>Owly</h1>

                <p className="tagline">
                    Your Personal Bookshelf Companion
                </p>

                <p className="description">
                    Scan your bookshelf to discover, organize and rediscover your books.
                </p>

            </div>

            <div className="actions">

                <button className="secondary-btn">
                    Upload Photo
                </button>

                <button className="primary-btn">
                    Scan Bookshelf
                </button>

            </div>

        </main>
    );
}
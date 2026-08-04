import "./AppShell.css";

export default function AppShell({ children }) {
    return (
        <div className="app-background">
            <div className="phone-frame">
                <div className="phone-content">{children}</div>
            </div>
        </div>
    );
}
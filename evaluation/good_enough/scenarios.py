"""Email scenarios dataset for Good Enough experiment.

This module defines diverse email scenarios across professional categories.
Each scenario includes context, recipient, goal, and required information
to enable objective quality evaluation.
"""

from dataclasses import dataclass, field


@dataclass
class EmailScenario:
    """A professional email scenario for the experiment.

    Attributes:
        id: Unique identifier
        category: Email category (meeting, request, apology, crisis, etc.)
        context: Background situation and context
        sender_role: Role of the email sender
        recipient: Who the email is addressed to
        goal: What the email should achieve
        key_info: Required information points that must be included
        tone_guidance: Expected tone for this scenario
        urgency: Urgency level ("low", "medium", "high", "critical")
        time_pressure: Why time matters (for crisis scenarios)
        max_iterations: Suggested max iterations before sending (None = no limit)
    """

    id: str
    category: str
    context: str
    sender_role: str
    recipient: str
    goal: str
    key_info: list[str] = field(default_factory=list)
    tone_guidance: str = "professional"
    urgency: str = "medium"  # low, medium, high, critical
    time_pressure: str = ""  # Why time matters
    max_iterations: int | None = None  # Suggested max iterations

    def to_prompt(self) -> str:
        """Convert scenario to a prompt for the email agent."""
        key_info_str = "\n".join(f"  - {info}" for info in self.key_info)

        # Add urgency context for high/critical scenarios
        urgency_str = ""
        if self.urgency in ("high", "critical"):
            urgency_str = f"\n**URGENCY:** {self.urgency.upper()}"
            if self.time_pressure:
                urgency_str += f" - {self.time_pressure}"

        return f"""Write a professional email based on this scenario:

**Your Role:** {self.sender_role}
**Recipient:** {self.recipient}
**Situation:** {self.context}
**Goal:** {self.goal}{urgency_str}

**Required Information to Include:**
{key_info_str}

**Tone:** {self.tone_guidance}

Write the email now (include Subject line):"""


# Curated scenarios across professional categories
SCENARIOS: list[EmailScenario] = [
    # === MEETING CATEGORY ===
    EmailScenario(
        id="meeting-01",
        category="meeting",
        context="You have a team standup scheduled for tomorrow at 10 AM, but you just learned you have a doctor's appointment at the same time that cannot be rescheduled.",
        sender_role="Software Engineer",
        recipient="Your team (5 people) and manager",
        goal="Reschedule the standup meeting to a different time",
        key_info=[
            "Current meeting time (tomorrow 10 AM)",
            "Reason for conflict (personal appointment)",
            "Proposed alternative times (2-3 options)",
            "Apology for the inconvenience",
        ],
        tone_guidance="apologetic but professional",
    ),
    EmailScenario(
        id="meeting-02",
        category="meeting",
        context="You need to schedule a project kickoff meeting with stakeholders from three different departments. The project is due to start next month.",
        sender_role="Project Manager",
        recipient="Representatives from Engineering, Marketing, and Finance",
        goal="Schedule a 1-hour kickoff meeting within the next two weeks",
        key_info=[
            "Purpose of the meeting (project kickoff)",
            "Expected duration (1 hour)",
            "Proposed dates/times (3 options)",
            "What attendees should prepare",
        ],
        tone_guidance="professional and organized",
    ),
    # === REQUEST CATEGORY ===
    EmailScenario(
        id="request-01",
        category="request",
        context="You're working on a report due Friday, but you need sales data from a colleague in another department who hasn't responded to your Slack messages.",
        sender_role="Business Analyst",
        recipient="Sales Team Lead",
        goal="Request Q3 sales data by Wednesday to complete your report",
        key_info=[
            "What data you need (Q3 sales figures)",
            "Why you need it (quarterly report)",
            "Deadline (Wednesday)",
            "Format preference (Excel or CSV)",
        ],
        tone_guidance="polite but clear about urgency",
    ),
    EmailScenario(
        id="request-02",
        category="request",
        context="Your company is implementing a new software tool and you need access credentials. IT has a ticket system but you need expedited access for a client demo tomorrow.",
        sender_role="Account Executive",
        recipient="IT Support",
        goal="Request expedited access to new CRM system",
        key_info=[
            "What access you need (CRM system)",
            "Why it's urgent (client demo tomorrow)",
            "Your employee ID for verification",
            "Preferred contact method for credentials",
        ],
        tone_guidance="urgent but respectful",
    ),
    # === APOLOGY CATEGORY ===
    EmailScenario(
        id="apology-01",
        category="apology",
        context="You promised to deliver a project milestone last Friday but encountered unexpected technical issues. The milestone is now 3 days late.",
        sender_role="Developer",
        recipient="Product Manager",
        goal="Apologize for the delay and provide updated timeline",
        key_info=[
            "What was delayed (milestone name)",
            "Reason for delay (technical issues)",
            "New delivery date",
            "Steps taken to prevent future delays",
        ],
        tone_guidance="sincere apology, solution-focused",
    ),
    EmailScenario(
        id="apology-02",
        category="apology",
        context="You accidentally sent confidential salary information to the wrong distribution list. You caught the error within an hour and need to address it.",
        sender_role="HR Coordinator",
        recipient="Department heads who received the email",
        goal="Apologize for the error and request deletion of the email",
        key_info=[
            "What happened (sent to wrong list)",
            "Request to delete the email",
            "Assurance of corrective measures",
            "Contact for questions",
        ],
        tone_guidance="serious, apologetic, reassuring",
    ),
    # === INTRODUCTION CATEGORY ===
    EmailScenario(
        id="intro-01",
        category="introduction",
        context="A new team member is joining your project next week. You're the tech lead and want to welcome them and provide initial onboarding information.",
        sender_role="Tech Lead",
        recipient="New team member (first day next Monday)",
        goal="Welcome them and provide essential first-day information",
        key_info=[
            "Warm welcome",
            "Your role and how you'll work together",
            "First day logistics (time, where to go)",
            "Offer to answer any questions",
        ],
        tone_guidance="warm, welcoming, helpful",
    ),
    EmailScenario(
        id="intro-02",
        category="introduction",
        context="You've been assigned to work with a client's team on a joint project. You need to introduce yourself and establish initial communication.",
        sender_role="Consultant",
        recipient="Client's project lead",
        goal="Introduce yourself and propose initial meeting",
        key_info=[
            "Your name and role",
            "Your relevant experience",
            "Excitement about the project",
            "Proposal for introductory call",
        ],
        tone_guidance="professional, enthusiastic, collaborative",
    ),
    # === FOLLOW-UP CATEGORY ===
    EmailScenario(
        id="followup-01",
        category="follow-up",
        context="You sent a proposal to a potential client 5 business days ago and haven't received a response. Your manager is asking for an update.",
        sender_role="Sales Representative",
        recipient="Potential client (VP of Operations)",
        goal="Follow up on proposal without being pushy",
        key_info=[
            "Reference to original proposal",
            "Offer to answer questions",
            "Availability for a call",
            "Next step if interested",
        ],
        tone_guidance="helpful, not pushy",
    ),
    EmailScenario(
        id="followup-02",
        category="follow-up",
        context="You interviewed for a position two weeks ago. The recruiter said they'd get back to you in a week, but you haven't heard anything.",
        sender_role="Job candidate",
        recipient="Recruiter",
        goal="Politely follow up on interview status",
        key_info=[
            "Position you interviewed for",
            "Interview date",
            "Continued interest in the role",
            "Availability for next steps",
        ],
        tone_guidance="professional, patient, interested",
    ),
    # === DECLINE CATEGORY ===
    EmailScenario(
        id="decline-01",
        category="decline",
        context="A colleague has asked you to present at an internal conference next month, but you're already overcommitted with a major product launch.",
        sender_role="Engineering Manager",
        recipient="Colleague organizing the conference",
        goal="Politely decline the speaking invitation",
        key_info=[
            "Appreciation for the invitation",
            "Reason for declining (launch commitment)",
            "Suggestion of alternative (colleague who could present)",
            "Openness to future opportunities",
        ],
        tone_guidance="appreciative, regretful, constructive",
    ),
    EmailScenario(
        id="decline-02",
        category="decline",
        context="A vendor has invited you to an expensive dinner to discuss their services. Your company policy prohibits accepting gifts over $50.",
        sender_role="Procurement Specialist",
        recipient="Vendor sales representative",
        goal="Decline the dinner invitation professionally",
        key_info=[
            "Thank them for the invitation",
            "Explain company policy (briefly)",
            "Offer alternative (coffee meeting, office call)",
            "Continued interest in their services",
        ],
        tone_guidance="gracious, firm, professional",
    ),
    # === THANK YOU CATEGORY ===
    EmailScenario(
        id="thanks-01",
        category="thank-you",
        context="A senior colleague mentored you through a difficult project that just launched successfully. They spent several hours helping you troubleshoot issues.",
        sender_role="Junior Developer",
        recipient="Senior colleague who mentored you",
        goal="Express genuine gratitude for their help",
        key_info=[
            "Specific things they helped with",
            "Impact of their help (successful launch)",
            "What you learned",
            "Offer to pay it forward",
        ],
        tone_guidance="heartfelt, specific, genuine",
    ),
    EmailScenario(
        id="thanks-02",
        category="thank-you",
        context="A customer gave you a glowing review that was shared company-wide. Their feedback helped you get recognition from leadership.",
        sender_role="Customer Success Manager",
        recipient="The customer who gave the review",
        goal="Thank them for their kind review",
        key_info=[
            "Reference to their review",
            "What it meant to you/team",
            "Commitment to continued excellent service",
            "Personal touch (something you enjoy about working with them)",
        ],
        tone_guidance="warm, professional, sincere",
    ),
    # === CLARIFICATION CATEGORY ===
    EmailScenario(
        id="clarify-01",
        category="clarification",
        context="You received requirements for a new feature, but some details are ambiguous. You need clarification before starting development.",
        sender_role="Software Developer",
        recipient="Product Owner",
        goal="Get clarification on feature requirements",
        key_info=[
            "Reference to the requirements document",
            "Specific questions (2-3 items)",
            "Your current understanding/assumptions",
            "Timeline impact if clarification is delayed",
        ],
        tone_guidance="clear, specific, collaborative",
    ),
    EmailScenario(
        id="clarify-02",
        category="clarification",
        context="You're processing an expense report but the receipts don't match the submitted amounts. You need to clarify before approving.",
        sender_role="Finance Associate",
        recipient="Employee who submitted expense report",
        goal="Request clarification on expense discrepancies",
        key_info=[
            "Which expenses have discrepancies",
            "The specific differences (numbers)",
            "What documentation is needed",
            "Deadline to respond to avoid processing delays",
        ],
        tone_guidance="professional, factual, helpful",
    ),
    # === ANNOUNCEMENT CATEGORY ===
    EmailScenario(
        id="announce-01",
        category="announcement",
        context="Your team has successfully launched a major product feature after 6 months of work. You want to announce this to the broader organization.",
        sender_role="Product Manager",
        recipient="All engineering and product staff",
        goal="Announce the feature launch and recognize the team",
        key_info=[
            "What launched (feature name/description)",
            "Key team members to recognize",
            "Business impact expected",
            "Where to learn more/provide feedback",
        ],
        tone_guidance="celebratory, inclusive, informative",
    ),
    EmailScenario(
        id="announce-02",
        category="announcement",
        context="You're leaving your current position for a new opportunity. You want to inform your immediate team and thank them.",
        sender_role="Departing employee",
        recipient="Immediate team (8 people)",
        goal="Announce departure and express gratitude",
        key_info=[
            "Last day at the company",
            "What you appreciated about working together",
            "Transition plan for your work",
            "Personal contact info to stay in touch",
        ],
        tone_guidance="warm, grateful, professional",
    ),
    # === UPDATE CATEGORY ===
    EmailScenario(
        id="update-01",
        category="update",
        context="You're leading a project and need to send a weekly status update to stakeholders. The project is on track with minor risks.",
        sender_role="Project Lead",
        recipient="Project stakeholders",
        goal="Provide clear weekly status update",
        key_info=[
            "Overall status (on track)",
            "Key accomplishments this week",
            "Upcoming milestones",
            "Risks and mitigations",
        ],
        tone_guidance="clear, concise, factual",
    ),
    EmailScenario(
        id="update-02",
        category="update",
        context="Your company's office is moving to a new location next month. You need to inform staff about logistics.",
        sender_role="Office Manager",
        recipient="All office employees",
        goal="Communicate office move details",
        key_info=[
            "New address",
            "Move date and transition timeline",
            "What employees need to do (pack personal items)",
            "Parking/transit information for new location",
        ],
        tone_guidance="clear, organized, helpful",
    ),
    # === CRISIS CATEGORY ===
    # These scenarios have HIGH urgency and time pressure
    # The key difference: waiting for "perfect" is WORSE than sending "good enough"
    EmailScenario(
        id="crisis-01",
        category="crisis",
        context="Your company discovered a data breach affecting 50,000 customer records 2 hours ago. Legal requires customer notification within 72 hours per GDPR, but your CEO wants to notify within 24 hours to maintain trust. You've been asked to draft the notification email immediately.",
        sender_role="Chief Communications Officer",
        recipient="All affected customers",
        goal="Notify customers of the data breach while maintaining trust",
        key_info=[
            "What happened (unauthorized access to customer database)",
            "What data was exposed (names, emails, encrypted passwords)",
            "What was NOT exposed (payment info, SSN)",
            "What we're doing about it (investigation, enhanced security)",
            "What customers should do (change passwords, monitor accounts)",
            "Contact for questions (dedicated support line)",
        ],
        tone_guidance="transparent, reassuring, action-oriented",
        urgency="critical",
        time_pressure="Must send within 4 hours. Every hour of delay increases regulatory and reputational risk.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-02",
        category="crisis",
        context="Your SaaS platform experienced a major outage affecting all customers for 45 minutes. Service is now restored but customers are flooding support with complaints. You need to send a status update immediately while the engineering team investigates root cause.",
        sender_role="VP of Customer Success",
        recipient="All customers (2,500 accounts)",
        goal="Acknowledge the outage, confirm restoration, and maintain customer confidence",
        key_info=[
            "Acknowledgment of the outage and impact",
            "Confirmation that service is restored",
            "Preliminary timeline (when it started, when it ended)",
            "Commitment to full root cause analysis",
            "Apology and next steps",
            "Where to get updates (status page URL)",
        ],
        tone_guidance="apologetic, transparent, professional",
        urgency="critical",
        time_pressure="Customers are actively complaining. Every minute without communication damages trust.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-03",
        category="crisis",
        context="A security researcher has responsibly disclosed a critical vulnerability in your product. They've given you 48 hours before public disclosure. You need to notify enterprise customers to update immediately while the patch is being deployed.",
        sender_role="Chief Security Officer",
        recipient="Enterprise customers (IT administrators)",
        goal="Urgent notification to update software before vulnerability is public",
        key_info=[
            "Severity of the vulnerability (critical)",
            "Affected versions",
            "Immediate action required (update to version X.Y.Z)",
            "Deadline (before public disclosure in 48 hours)",
            "How to update (steps or link to documentation)",
            "Support contact for urgent assistance",
        ],
        tone_guidance="urgent, clear, technical but accessible",
        urgency="critical",
        time_pressure="48-hour countdown to public disclosure. Unpatched systems will be at risk.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-04",
        category="crisis",
        context="Your company's CEO made controversial comments in an interview that went viral. Social media is exploding with criticism. HR is being flooded with employee concerns. You need to send an internal communication before the all-hands meeting in 2 hours.",
        sender_role="Chief People Officer",
        recipient="All employees",
        goal="Address employee concerns and provide context before all-hands meeting",
        key_info=[
            "Acknowledgment of the situation",
            "Commitment to company values",
            "What leadership is doing (CEO will address at all-hands)",
            "Reminder of internal support resources",
            "Request for patience and unity",
            "When/where the all-hands meeting will be",
        ],
        tone_guidance="empathetic, calm, unifying",
        urgency="high",
        time_pressure="All-hands meeting in 2 hours. Employees need context before then.",
        max_iterations=3,
    ),
    EmailScenario(
        id="crisis-05",
        category="crisis",
        context="Your manufacturing facility discovered a potential safety defect in products shipped last month. While not yet confirmed, legal advises proactive customer notification. A recall may follow, but immediate communication is needed while investigation continues.",
        sender_role="Director of Product Safety",
        recipient="Customers who purchased affected product batch",
        goal="Proactive safety notification while investigation continues",
        key_info=[
            "What product/batch is potentially affected",
            "Nature of the potential safety concern",
            "Precautionary advice (stop using, return, wait for update)",
            "Status of investigation",
            "How to check if their product is affected",
            "Contact for questions and returns",
        ],
        tone_guidance="cautious, safety-first, transparent",
        urgency="high",
        time_pressure="Legal liability increases with delay. Proactive notification is better than reactive recall.",
        max_iterations=3,
    ),
    EmailScenario(
        id="crisis-06",
        category="crisis",
        context="Your company's payment processor is experiencing issues. Customers are seeing failed transactions and duplicate charges. Finance is working on reversals, but customer-facing communication is needed NOW while the fix is in progress.",
        sender_role="Head of Customer Experience",
        recipient="Customers who transacted in the last 4 hours",
        goal="Acknowledge payment issues and provide immediate guidance",
        key_info=[
            "Acknowledgment of payment processing issues",
            "Types of issues (failed transactions, potential duplicates)",
            "Assurance that no charges are lost and duplicates will be reversed",
            "Advice (don't retry, wait for confirmation)",
            "Expected resolution timeline",
            "How affected customers will be contacted",
        ],
        tone_guidance="calm, reassuring, solution-focused",
        urgency="critical",
        time_pressure="Customers are panicking about money. Every minute of silence increases support volume.",
        max_iterations=2,
    ),
    # === ADDITIONAL CRISIS SCENARIOS FOR STATISTICAL POWER ===
    # Healthcare domain
    EmailScenario(
        id="crisis-07",
        category="crisis",
        context="Your hospital's electronic health records system crashed during a busy shift. Staff are switching to paper records, but patients and families need to be informed about potential delays in care coordination.",
        sender_role="Chief Medical Officer",
        recipient="All patients currently admitted and their families",
        goal="Inform patients about EHR outage and reassure about care continuity",
        key_info=[
            "EHR system is temporarily unavailable",
            "All critical care continues with paper backup protocols",
            "Medication schedules are being maintained manually",
            "Expected system restoration timeline",
            "What patients should do if they have concerns",
            "Direct line to patient advocacy for questions",
        ],
        tone_guidance="calm, medically professional, reassuring",
        urgency="critical",
        time_pressure="Patients are anxious about their care. Clear communication prevents panic.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-08",
        category="crisis",
        context="A medication contamination issue has been identified affecting prescriptions filled in the last week. While no adverse events reported yet, patients need to be notified immediately to stop taking the affected medication.",
        sender_role="Chief Pharmacy Officer",
        recipient="Patients who received affected prescription",
        goal="Urgent notification to stop medication and seek replacement",
        key_info=[
            "Which medication batch is affected",
            "Immediate action: STOP taking this medication",
            "Symptoms to watch for",
            "How to get replacement prescription",
            "24/7 pharmacist hotline for questions",
            "Assurance about covering replacement costs",
        ],
        tone_guidance="urgent, safety-focused, clear instructions",
        urgency="critical",
        time_pressure="Patient safety is paramount. Every hour of delay increases risk of harm.",
        max_iterations=2,
    ),
    # Financial/Legal domain
    EmailScenario(
        id="crisis-09",
        category="crisis",
        context="Your fintech company discovered unauthorized trading activity in several customer accounts. Accounts have been frozen, but customers need immediate notification about the security incident and next steps.",
        sender_role="Chief Compliance Officer",
        recipient="Affected account holders",
        goal="Notify customers about security incident and frozen accounts",
        key_info=[
            "Unauthorized activity was detected in your account",
            "Account has been temporarily frozen for protection",
            "No customer funds have been lost (all trades reversed)",
            "Identity verification process to restore access",
            "Regulatory notification has been filed",
            "Dedicated support team contact for affected customers",
        ],
        tone_guidance="serious, protective, action-oriented",
        urgency="critical",
        time_pressure="Customers will panic when they see frozen accounts. Proactive communication is essential.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-10",
        category="crisis",
        context="Your company received a regulatory cease-and-desist order for a product feature that was operating in a legal grey area. The feature must be disabled within 24 hours. Customers actively using this feature need immediate notification.",
        sender_role="General Counsel",
        recipient="Customers using affected feature",
        goal="Notify customers about feature discontinuation and alternatives",
        key_info=[
            "Specific feature being discontinued",
            "Regulatory reason (briefly, without admitting wrongdoing)",
            "Timeline: feature disabled in 24 hours",
            "How to export/migrate any data from the feature",
            "Alternative solutions or workarounds",
            "Refund policy for any paid aspects of the feature",
        ],
        tone_guidance="professional, factual, solution-focused",
        urgency="high",
        time_pressure="24-hour regulatory deadline. Customers need time to migrate.",
        max_iterations=3,
    ),
    # Infrastructure/Operations domain
    EmailScenario(
        id="crisis-11",
        category="crisis",
        context="Your data center experienced a fire in one wing. No injuries, but multiple client servers are offline. Some data may need to be restored from backups. Enterprise clients need immediate status updates.",
        sender_role="VP of Data Center Operations",
        recipient="Enterprise clients with affected infrastructure",
        goal="Notify clients about incident and recovery status",
        key_info=[
            "Fire incident occurred, no personnel injuries",
            "Which systems/racks are affected",
            "Current status of their specific services",
            "Data protection status (backups available from X date)",
            "Expected recovery timeline by tier",
            "Dedicated incident manager assigned to their account",
        ],
        tone_guidance="transparent, technical, accountable",
        urgency="critical",
        time_pressure="Clients' businesses are impacted. They need immediate clarity for their own crisis response.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-12",
        category="crisis",
        context="A severe weather warning has been issued. Your company's main office and warehouse will need to close early today, and potentially remain closed tomorrow. Employees, customers with pending orders, and delivery partners need notification.",
        sender_role="Operations Director",
        recipient="All employees, pending order customers, and logistics partners",
        goal="Announce emergency closure and provide guidance",
        key_info=[
            "Office and warehouse closing at 2 PM today",
            "Status for tomorrow (monitoring, decision by 6 AM)",
            "Employee safety instructions",
            "Impact on pending orders and deliveries",
            "How customers can track order updates",
            "Emergency contact for critical issues",
        ],
        tone_guidance="calm, safety-focused, organized",
        urgency="high",
        time_pressure="People need time to prepare. Delayed notification causes scrambling.",
        max_iterations=3,
    ),
    # Supply Chain domain
    EmailScenario(
        id="crisis-13",
        category="crisis",
        context="Your primary supplier's factory was damaged by flooding. Lead times will increase from 2 weeks to 8 weeks minimum. Your customers with pending orders and long-term contracts need immediate notification.",
        sender_role="VP of Supply Chain",
        recipient="Customers with pending orders",
        goal="Notify about supply disruption and mitigation steps",
        key_info=[
            "Supplier disruption affecting product availability",
            "New expected lead times",
            "Options: wait, partial shipment, alternative product",
            "How we're mitigating (backup suppliers, expedited production)",
            "Contract implications and any penalties waived",
            "Dedicated account manager for affected orders",
        ],
        tone_guidance="transparent, solution-oriented, professional",
        urgency="high",
        time_pressure="Customers need to adjust their own plans. Early notice allows them to find alternatives.",
        max_iterations=3,
    ),
    EmailScenario(
        id="crisis-14",
        category="crisis",
        context="Port congestion has delayed a critical shipment of holiday inventory. Your retail partners expecting inventory next week won't receive it until after Black Friday. This threatens their holiday season.",
        sender_role="Head of Retail Partnerships",
        recipient="Retail partners expecting delayed shipment",
        goal="Notify about delay and offer mitigation support",
        key_info=[
            "Shipment delayed due to port congestion",
            "New expected arrival date (after Black Friday)",
            "Products affected and quantities",
            "Mitigation: air freight partial shipment option",
            "Marketing support to offset impact",
            "Credit terms adjustment offer",
        ],
        tone_guidance="apologetic, partnership-focused, solution-oriented",
        urgency="critical",
        time_pressure="Retail partners need to adjust holiday plans. Every day of delay reduces their options.",
        max_iterations=2,
    ),
    # Cybersecurity domain
    EmailScenario(
        id="crisis-15",
        category="crisis",
        context="Your security team detected an active ransomware attack. The attack was contained but some customer-facing systems are offline. Customers are seeing error messages and need immediate explanation.",
        sender_role="Chief Information Security Officer",
        recipient="All customers",
        goal="Explain service disruption and security measures being taken",
        key_info=[
            "We detected and contained a cybersecurity incident",
            "Some services are temporarily offline",
            "No evidence of customer data exfiltration",
            "What we're doing to restore services",
            "Expected restoration timeline",
            "How to verify legitimate communications from us",
        ],
        tone_guidance="transparent, security-conscious, reassuring",
        urgency="critical",
        time_pressure="Customers seeing errors may assume the worst. Clear communication prevents speculation.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-16",
        category="crisis",
        context="An employee laptop with customer data was stolen from a car. While encrypted, your data protection policy requires notification. You don't believe data was accessed but must inform affected customers.",
        sender_role="Data Protection Officer",
        recipient="Customers whose data was on the device",
        goal="Transparent notification of potential data exposure",
        key_info=[
            "A company laptop was stolen",
            "Your data may have been on this device",
            "What data was potentially exposed",
            "Device was encrypted (low risk of access)",
            "Steps we're taking (remote wipe attempted, investigation)",
            "Precautionary steps you can take",
        ],
        tone_guidance="transparent, apologetic, informative",
        urgency="high",
        time_pressure="Regulatory requirement to notify within 72 hours. Proactive notification is better.",
        max_iterations=3,
    ),
    # HR/Internal domain
    EmailScenario(
        id="crisis-17",
        category="crisis",
        context="Your company is implementing immediate layoffs affecting 15% of staff. Affected employees are being notified in individual meetings today. The remaining employees need communication explaining the situation before rumors spread.",
        sender_role="Chief Human Resources Officer",
        recipient="All remaining employees",
        goal="Announce reduction in force with transparency and compassion",
        key_info=[
            "Company is reducing workforce by 15%",
            "Business reason (briefly)",
            "Affected colleagues are being notified individually today",
            "Support being provided to departing colleagues",
            "What this means for remaining employees",
            "Leadership Q&A session scheduled for this afternoon",
        ],
        tone_guidance="compassionate, transparent, leadership-focused",
        urgency="critical",
        time_pressure="Rumors spread fast. Official communication must come before the grapevine.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-18",
        category="crisis",
        context="A workplace harassment complaint has become public through social media. Employees are concerned and demanding a response. You need to address the situation without compromising the investigation.",
        sender_role="Chief Ethics Officer",
        recipient="All employees",
        goal="Address concerns while protecting investigation integrity",
        key_info=[
            "Awareness of concerns raised publicly",
            "Commitment to taking all concerns seriously",
            "Independent investigation is underway",
            "Cannot share details to protect all parties",
            "Resources for anyone experiencing or witnessing harassment",
            "Commitment to communication when investigation concludes",
        ],
        tone_guidance="serious, supportive, legally careful",
        urgency="high",
        time_pressure="Social media discussion is active. Silence is interpreted as indifference.",
        max_iterations=3,
    ),
    # Customer Service domain
    EmailScenario(
        id="crisis-19",
        category="crisis",
        context="A viral social media post claims your product caused injury. While investigating (and product seems safe), negative posts are spreading rapidly. Customers are demanding information.",
        sender_role="VP of Customer Safety",
        recipient="All customers",
        goal="Address viral safety concern while investigation continues",
        key_info=[
            "Awareness of social media concerns",
            "Immediate investigation launched",
            "Current safety record and testing protocols",
            "Steps customers should take if concerned",
            "How to report any issues directly to us",
            "Commitment to transparency when investigation concludes",
        ],
        tone_guidance="safety-focused, measured, transparent",
        urgency="high",
        time_pressure="Social media moves fast. Every hour of silence allows the narrative to solidify.",
        max_iterations=3,
    ),
    EmailScenario(
        id="crisis-20",
        category="crisis",
        context="Your airline cancelled 200 flights due to a pilot scheduling software glitch. Thousands of passengers are stranded. The booking system is overloaded. Passengers need immediate guidance.",
        sender_role="Chief Customer Officer",
        recipient="Passengers on cancelled flights",
        goal="Provide immediate guidance and rebooking options",
        key_info=[
            "Your flight has been cancelled",
            "Cause: operational scheduling issue (not safety-related)",
            "Automatic rebooking on next available flight",
            "Hotel and meal vouchers if overnight delay",
            "How to check your new booking",
            "Full refund option if you choose not to travel",
        ],
        tone_guidance="apologetic, practical, action-oriented",
        urgency="critical",
        time_pressure="Stranded passengers need immediate answers. Airport staff are overwhelmed.",
        max_iterations=2,
    ),
    # Environmental/Compliance domain
    EmailScenario(
        id="crisis-21",
        category="crisis",
        context="Your factory had an accidental chemical release that briefly affected air quality in the surrounding neighborhood. While contained quickly, residents are concerned and regulators are on site.",
        sender_role="Environmental Health & Safety Director",
        recipient="Surrounding community residents",
        goal="Inform community about incident and safety measures",
        key_info=[
            "Brief chemical release occurred at our facility",
            "Release was contained within 30 minutes",
            "Current air quality readings are normal",
            "What chemical was involved and health implications",
            "Who to contact if experiencing symptoms",
            "Regulatory investigation is underway",
        ],
        tone_guidance="transparent, community-focused, safety-first",
        urgency="critical",
        time_pressure="Community trust erodes with silence. Proactive communication is essential.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-22",
        category="crisis",
        context="An audit discovered your company has been inadvertently violating accessibility regulations on your website. You have 30 days to achieve compliance or face fines. Affected users need notification.",
        sender_role="Chief Accessibility Officer",
        recipient="Users who use assistive technologies",
        goal="Acknowledge accessibility issues and communicate remediation plan",
        key_info=[
            "We identified accessibility issues on our platform",
            "Sincere apology for any difficulties experienced",
            "Specific issues being addressed",
            "Timeline for remediation (features coming when)",
            "Interim alternatives for affected features",
            "Dedicated feedback channel for accessibility concerns",
        ],
        tone_guidance="humble, committed, user-focused",
        urgency="high",
        time_pressure="30-day compliance deadline. Users deserve immediate acknowledgment.",
        max_iterations=3,
    ),
    # Partner/B2B domain
    EmailScenario(
        id="crisis-23",
        category="crisis",
        context="Your API suffered a silent failure that returned incorrect data to partners for 6 hours before detection. Partners may have made business decisions based on bad data. Immediate notification is required.",
        sender_role="VP of Platform Engineering",
        recipient="API partners and developers",
        goal="Notify about data integrity issue and provide recovery guidance",
        key_info=[
            "API data integrity issue occurred between X and Y times",
            "Affected endpoints and data types",
            "How to identify affected transactions",
            "Data reconciliation support we're providing",
            "Root cause (briefly) and prevention measures",
            "SLA credits or compensation process",
        ],
        tone_guidance="technical, accountable, solution-focused",
        urgency="critical",
        time_pressure="Partners may be compounding errors. Immediate notification stops the bleeding.",
        max_iterations=2,
    ),
    EmailScenario(
        id="crisis-24",
        category="crisis",
        context="Your major enterprise software update introduced a critical bug that's corrupting customer databases. You've identified the issue and have a fix, but customers need to stop using the affected feature immediately.",
        sender_role="VP of Engineering",
        recipient="Enterprise customers on affected version",
        goal="Emergency notification to stop using affected feature",
        key_info=[
            "CRITICAL: Stop using [feature] immediately",
            "Bug in version X.Y.Z may corrupt data",
            "Which operations trigger the bug",
            "How to check if your data is affected",
            "Patch available: download link and instructions",
            "Data recovery support for affected customers",
        ],
        tone_guidance="urgent, clear, technical",
        urgency="critical",
        time_pressure="Every use of the feature causes more damage. Immediate action required.",
        max_iterations=2,
    ),
]


# Crisis scenarios as a separate list for focused experiments
CRISIS_SCENARIOS: list[EmailScenario] = [s for s in SCENARIOS if s.category == "crisis"]


def load_scenarios(
    categories: list[str] | None = None,
    limit: int | None = None,
    random_seed: int | None = None,
) -> list[EmailScenario]:
    """Load email scenarios with optional filtering.

    Args:
        categories: Filter to specific categories (None = all)
        limit: Maximum number of scenarios to return
        random_seed: If provided, randomly sample scenarios

    Returns:
        List of EmailScenario objects
    """
    scenarios = SCENARIOS.copy()

    if categories:
        scenarios = [s for s in scenarios if s.category in categories]

    if random_seed is not None:
        import random

        rng = random.Random(random_seed)
        rng.shuffle(scenarios)

    if limit:
        scenarios = scenarios[:limit]

    return scenarios


def get_scenario_by_id(scenario_id: str) -> EmailScenario | None:
    """Get a specific scenario by ID."""
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    return None


def list_categories() -> list[str]:
    """Get list of all scenario categories."""
    return sorted({s.category for s in SCENARIOS})

module.exports = async (project_name, expirationDays = 30) => {
    const { CF_ACCOUNT_ID, CF_TOKEN } = process.env;

    const endpoint = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/pages/projects/${project_name}/deployments`;

    const headers = {
        "Content-Type": "application/json;charset=UTF-8",
        Authorization: `Bearer ${CF_TOKEN}`,
    };

    let page = 1;
    let totalPages = 1;
    const deploymentIds = [];

    do {
        // Get the list of deployments
        const response = await fetch(`${endpoint}?page=${page}`, {
            headers,
        });
        const { result, result_info } = await response.json();
        // Set the total pages
        if (page == 1) {
            totalPages = result_info.total_pages;
        }

        console.log(`Found ${result.length} deployments on page ${page}/${totalPages}.`);

        for (const deployment of result) {
            // Check if the deployment was created within the last x days
            if ((Date.now() - new Date(deployment.created_on)) / 86400000 > expirationDays) {
                console.log(`Queueing deployment ${deployment.id} created on ${deployment.created_on} for deletion`);
                deploymentIds.push(deployment.id);
            }
        }

        // Increment page
        page++;
    } while (page <= totalPages);

    // Delete
    for (const [i, deployment_id] of deploymentIds.entries()) {
        console.log(`Deleting deployment ${deployment_id} (${i + 1}/${deploymentIds.length})`);

        // Delete the deployment
        await fetch(`${endpoint}/${deployment_id}`, {
            method: "DELETE",
            headers,
        });
    }
};
